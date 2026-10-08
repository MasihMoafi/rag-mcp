"""Document ingestion, source locations, and persisted metadata lifecycle."""

import json
import os
import sys
from typing import Dict, List
from utils.persistence import write_json_atomic

from .chunking import chunk_document_spans
from .contextual_retrieval import select_document_context
from .formats import INDEXABLE_EXTENSIONS
from .index_lifecycle import RebuildRequired
from .parsers.text import clean_text
from . import tracing


def locate_chunk(self, full_text: str, content: str, ext: str, search_from: int) -> tuple:
    """Recover a human-readable location for a chunk: a page range for PDFs
    (pages joined with \\f during extraction), a cell number for notebooks
    (cells joined with \\f, including ranges), or a line range for everything else, where
    newlines survive extraction untouched. Returns (location_string, next
    search_from) so callers can scan the document once, left to right."""
    start = full_text.find(content, search_from)
    if start == -1:
        start = full_text.find(content)
    if start == -1:
        return ("", search_from)

    if ext in {".pdf", ".ipynb"}:
        first = full_text.count("\f", 0, start) + 1
        last = first + content.count("\f")
        prefix = "p." if ext == ".pdf" else "cell "
        location = f"{prefix}{first}" if first == last else f"{prefix}{first}-{last}"
    else:
        line_start = full_text.count("\n", 0, start) + 1
        line_end = line_start + content.count("\n")
        location = f"L{line_start}" if line_end == line_start else f"L{line_start}-L{line_end}"
    return (location, start + 1)

def add_chunk_context(self, chunk: str, document_context: str) -> str:
    if not self.config.get("use_contextual", False):
        return chunk

    try:
        context = self.contextualizer(chunk, document_context).strip()
    except Exception as error:
        print(
            f"[RAG V2] Contextual retrieval failed; keeping the original chunk: {type(error).__name__}",
            file=sys.stderr,
        )
        return chunk
    return f"{context}\n{chunk}" if context else chunk

def load_or_create_database(self):
    """Load or create the persisted database."""
    persist_dir = self.config.get("persist_dir")

    if not os.path.exists(persist_dir) or self.config.get("force_reindex", False):
        print(f"[RAG V2] Creating new database at {persist_dir}...")
        self._index_documents()
    elif self.config.get("source_delta"):
        print(f"[RAG V2] Refreshing changed sources in {persist_dir}...")
        self._refresh_documents(self.config["source_delta"])
    else:
        print(f"[RAG V2] Loading existing database from {persist_dir}...")
        self._load_existing_database()

def source_files(self):
    """The files of this scope, in the order the index reads them."""
    files_to_process = []
    for path in self.config.get("document_paths"):
        if not os.path.exists(path):
            continue

        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.lower().endswith(INDEXABLE_EXTENSIONS):
                    files_to_process.append(os.path.join(path, file))
        else:
            files_to_process.append(path)
    return files_to_process

_PROSE_EXTENSIONS = (".md", ".markdown", ".txt", ".pdf", ".docx")


def build_documents(self, files_to_process):
    """Extract, chunk and optionally contextualize the given sources."""
    import concurrent.futures
    all_docs = []
    document_contents = {}

    def process_file(file_path):
        content = self._extract_content(file_path)
        if content:
            return file_path, content
        return None

    # First pass: extract all document content using ThreadPoolExecutor
    with tracing.span('ingestion.parse', input_count=len(files_to_process)):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_file, files_to_process)
            for result in results:
                if result:
                    file_path, content = result
                    document_contents[file_path] = content
        tracing.annotate(output_count=len(document_contents))

    # Second pass: chunk and optionally add context
    for file_path, full_content in document_contents.items():
        print(f"[RAG V2] Processing {os.path.basename(file_path)}...")

        with tracing.span('ingestion.chunk', source_id=tracing.identity(file_path)):
            chunks = chunk_document_spans(
                full_content,
                file_path=file_path,
                chunk_size=self.config.get("chunk_size"),
                chunk_overlap=self.config.get("chunk_overlap")
            )
            tracing.annotate(output_count=len(chunks))

        ext = os.path.splitext(file_path)[1].lower()
        raw_chunks = [item["content"].replace("\f", " ") for item in chunks]
        for i, item in enumerate(chunks):
            chunk = item["content"]
            if chunk.strip():
                location, _ = self._locate_chunk(full_content, chunk, ext, item["start_offset"])
                clean_chunk = raw_chunks[i]
                # Search prose without converter noise; citations keep the exact source text.
                searchable = (clean_text(clean_chunk, markdown=ext in (".md", ".markdown"))
                              if ext in _PROSE_EXTENSIONS else clean_chunk)
                if not searchable:
                    continue

                document_context = full_content
                if self.config.get("use_contextual", False):
                    document_context = select_document_context(
                        full_content,
                        raw_chunks,
                        i,
                        max_source_chars=int(self.config.get("contextual_source_max_chars", 12000)),
                        start_chunks=int(self.config.get("contextual_start_chunks", 2)),
                        previous_chunks=int(self.config.get("contextual_prev_chunks", 2)),
                    )
                contextualized_chunk = self._add_context_to_chunk(searchable, document_context)

                doc = {
                    "content": contextualized_chunk,
                    "original_content": clean_chunk,  # Keep original for comparison
                    "source": file_path,
                    "file_path": file_path,
                    "location": location,
                    "start_line": int(location[1:].split("-")[0]) if location.startswith("L") else 0,
                    "end_line": int(location.rsplit("L", 1)[1]) if location.startswith("L") else 0,
                    "page_number": int(location[2:].split("-")[0]) if location.startswith("p.") else 0,
                    "heading": item["heading"],
                    "start_offset": item["start_offset"],
                    "end_offset": item["end_offset"],
                    "chunk_id": i,
                    "id": f"{os.path.realpath(file_path)}::chunk:{i}"
                }
                all_docs.append(doc)

    return all_docs


@tracing.traced('index.build')
def index_documents(self):
    """Extract, chunk, optionally contextualize, and persist documents."""
    all_docs = build_documents(self, source_files(self))

    if not all_docs:
        raise ValueError("No readable text found in the selected files.")

    # Add documents to indexes
    print(f"[RAG V2] Indexing {len(all_docs)} chunks...")

    self.vector_db.add_documents_batch(all_docs, batch_size=512)


    if self.bm25_index is not None:
        print(f"[RAG V2] Building BM25 index...")
        for doc in all_docs:
            self.bm25_index.add_document(doc)
        self.bm25_index._build_index()

    # Save database
    self._save_database(all_docs)
    print(f"[RAG V2] Database created with {len(all_docs)} chunks")


@tracing.traced('index.refresh')
def refresh_documents(self, delta):
    """Re-read only the changed sources; keep the stored embeddings of the rest."""
    changed = {os.path.realpath(path) for path in delta["changed"]}
    stale = changed | {os.path.realpath(path) for path in delta["removed"]}
    fresh = lambda doc: os.path.realpath(doc.get("file_path", "")) not in stale

    kept = [doc for doc in read_documents(self) if fresh(doc)]
    reused = [record for record in self.vector_db.stored_documents() if fresh(record)]
    refreshed = build_documents(self, [path for path in source_files(self)
                                       if os.path.realpath(path) in changed])
    if not kept and not refreshed:
        raise ValueError("No readable text found in the selected files.")

    print(f"[RAG V2] Re-indexing {len(refreshed)} chunks from {len(changed)} changed sources...")
    self.vector_db.add_documents_batch(reused + refreshed, batch_size=512)

    if self.bm25_index is not None:
        for doc in kept + refreshed:
            self.bm25_index.add_document(doc)
        self.bm25_index._build_index()

    self._save_database(kept + refreshed)
    print(f"[RAG V2] Database refreshed with {len(kept) + len(refreshed)} chunks")


@tracing.traced('index.persist_metadata')
def save_documents(self, docs: List[Dict]):
    """Persist source records for reopen and lexical-index reconstruction."""
    persist_dir = self.config.get("persist_dir")
    os.makedirs(persist_dir, exist_ok=True)
    
    write_json_atomic(os.path.join(persist_dir, "docs.json"), docs)

def read_documents(self):
    """Persisted passages, or a rebuild demand when they cannot be trusted."""
    persist_dir = self.config.get("persist_dir")
    docs_file = os.path.join(persist_dir, "docs.json")

    db_count = self.vector_db.count()

    if db_count <= 0 or not os.path.exists(docs_file):
        raise RebuildRequired("Database or document metadata is missing")
    try:
        with open(docs_file, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        if not isinstance(docs, list) or len(docs) != db_count or any(
            not isinstance(doc, dict) or not isinstance(doc.get('content'), str)
            for doc in docs
        ):
            raise ValueError("Metadata does not match stored documents")
    except (OSError, ValueError) as error:
        print(f"[RAG] Cannot reuse document metadata; rebuilding from sources: {error}", file=sys.stderr)
        raise RebuildRequired("Document metadata needs recovery") from error
    return docs

def load_existing_database(self):
    """Load existing database"""
    docs = read_documents(self)

    if self.bm25_index is not None:
        print(f"[RAG V2] Loading {len(docs)} chunks and rebuilding BM25...")
        for doc in docs:
            self.bm25_index.add_document(doc)
        self.bm25_index._build_index()

    print(f"[RAG V2] Loaded {len(docs)} chunks from existing database ({self.backend_type}: {self.vector_db.count()})")
