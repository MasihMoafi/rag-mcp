import sys
import os, re, json, math
try:
    import torch
except ImportError:  # light install without PyTorch: CPU only, no sentence-transformers
    torch = None
from typing import List, Dict, Tuple, Any

# --- FIX for Ollama Proxy ---
# This is necessary to ensure the local Ollama server can be reached.
def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]
clear_proxy_settings()

from collections import Counter
from .embeddings import build_embedder, probe_dimensions
from .contextual_retrieval import AnthropicContextualizer, select_document_context
from .lancedb_backend import LanceDBBackend
from . import parsing
from .formats import INDEXABLE_EXTENSIONS


def select_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps and mps.is_available() else "cpu"


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 100) -> List[str]:
    """Return contiguous source slices, preferring structural/word boundaries.

    Sizes are character budgets. Only a token longer than the budget requires
    a hard cut; overlap is selected from the source, never stitched onto a chunk.
    """
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    if not text or not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            # Avoid tiny chunks when a heading/paragraph ends near the start.
            minimum = len(window) // 2
            for pattern in (r"\n\s*\n", r"[.!?؟](?=\s)", r"\n", r"\s+"):
                boundaries = [match.end() for match in re.finditer(pattern, window)
                              if match.end() >= minimum]
                if boundaries:
                    end = start + boundaries[-1]
                    break
            else:
                # A short prefix before a long word still provides a valid
                # boundary; do not split that word just to fill half a chunk.
                spaces = list(re.finditer(r"\s+", window))
                if spaces:
                    end = start + spaces[-1].end()
        chunk = text[start:end].strip()

        if chunk.strip():
            chunks.append(chunk)

        if end == len(text):
            break
        next_start = max(start + 1, end - chunk_overlap)
        # Do not prepend a sentence/paragraph fragment merely to meet the
        # overlap budget. Prefer a complete unit within that budget, even when
        # this means no overlap at an existing paragraph boundary.
        suffix = text[next_start:end]
        for pattern in (r"\n\s*\n", r"[.!?؟](?=\s)"):
            boundary = re.search(pattern, suffix)
            if boundary:
                next_start += boundary.end()
                break
        # Move an overlap boundary out of a word. For an oversized unbroken
        # token retain the hard-window fallback rather than dropping its tail.
        candidate = next_start
        while candidate < end and candidate > 0 and not text[candidate - 1].isspace():
            candidate += 1
        start = candidate if candidate < end else end

    return chunks


_MD_EXTS = {".md", ".markdown"}
_CODE_EXTS = {
    ".py", ".rs", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".cpp", ".hpp",
    ".go", ".java", ".kt", ".sh", ".bash", ".zsh", ".sql",
}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
_CODE_SYMBOL_RE = re.compile(
    r"^\s*(?:pub\s+|pub\(crate\)\s+|async\s+|export\s+|static\s+)*"
    r"(?:def|fn|class|function|struct|impl|interface|type)\s+([A-Za-z_][A-Za-z0-9_]*)"
)


def _split_markdown_sections(text: str) -> List[Tuple[str, str]]:
    """Split markdown into (heading_breadcrumb, section_text) pairs on '#' headings."""
    sections: List[Tuple[str, str]] = []
    stack: List[Tuple[int, str]] = []
    current_lines: List[str] = []

    def flush():
        if current_lines:
            breadcrumb = " > ".join(title for _, title in stack)
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((breadcrumb, body))

    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            flush()
            current_lines = [line]
            level = len(match.group(1))
            title = match.group(2).strip()
            stack = [entry for entry in stack if entry[0] < level]
            stack.append((level, title))
        else:
            current_lines.append(line)
    flush()

    return sections


def _nearest_code_symbol(text: str) -> str:
    """Best-effort function/class name a code chunk falls under, for citations."""
    for line in text.split("\n"):
        match = _CODE_SYMBOL_RE.match(line)
        if match:
            return match.group(1)
    return ""


def chunk_document(
    text: str, file_path: str = "", chunk_size: int = 700, chunk_overlap: int = 100
) -> List[Dict[str, str]]:
    """Source-faithful chunks with Markdown section boundaries and preferred
    paragraph, sentence, and word boundaries within the character budget.
    Oversized unbroken tokens fall back to hard cuts. Returns a "heading"
    breadcrumb (markdown section path or nearest code symbol) for citations.
    """
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    if not text or not text.strip():
        return []

    ext = os.path.splitext(file_path)[1].lower()

    if ext in _MD_EXTS and _HEADING_RE.search(text):
        results = []
        for breadcrumb, section in _split_markdown_sections(text):
            for chunk in chunk_text(section, chunk_size, chunk_overlap):
                if chunk.strip():
                    results.append({"content": chunk, "heading": breadcrumb})
        if results:
            return results

    heading_fn = _nearest_code_symbol if ext in _CODE_EXTS else (lambda _c: "")
    return [{"content": chunk, "heading": heading_fn(chunk)}
            for chunk in chunk_text(text, chunk_size, chunk_overlap)]



class BM25Index:
    """
    Custom BM25 implementation from RAG Course.
    Uses proper IDF calculation and BM25 scoring formula.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer=None):
        self.documents: List[Dict[str, Any]] = []
        self._corpus_term_counts: List[Counter] = []
        self._doc_len: List[int] = []
        self._doc_freqs: Dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._idf: Dict[str, float] = {}
        self._index_built: bool = False

        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter
        self._tokenizer = tokenizer if tokenizer else self._default_tokenizer

    def _default_tokenizer(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.split(r"\W+", text)
        return [token for token in tokens if token]

    def _calculate_idf(self):
        """Calculate Inverse Document Frequency for each term"""
        N = len(self.documents)
        self._idf = {}
        for term, freq in self._doc_freqs.items():
            # Standard BM25 IDF formula.
            idf_score = math.log(((N - freq + 0.5) / (freq + 0.5)) + 1)
            self._idf[term] = idf_score

    def add_document(self, document: Dict[str, Any]):
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        content = document.get("content", "")
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        doc_tokens = self._tokenizer(content)
        self.documents.append(document)
        self._corpus_term_counts.append(Counter(doc_tokens))
        
        # Update statistics
        self._doc_len.append(len(doc_tokens))
        for token in self._corpus_term_counts[-1]:
            self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
        
        self._index_built = False

    def _build_index(self):
        """Build the BM25 index by calculating average doc length and IDF scores"""
        if not self.documents:
            self._avg_doc_len = 0.0
            self._idf = {}
            self._index_built = True
            return

        self._avg_doc_len = sum(self._doc_len) / len(self.documents)
        self._calculate_idf()
        self._index_built = True

    def search(self, query: str, k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        if not self.documents:
            return []

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        if not self._index_built:
            self._build_index()

        if self._avg_doc_len == 0:
            return []

        query_tokens = self._tokenizer(query)
        if not query_tokens:
            return []

        # Calculate BM25 scores for all documents
        scores = []
        for i, doc in enumerate(self.documents):
            score = self._compute_bm25_score(query_tokens, i)
            if score > 1e-9:  # Only include documents with non-zero scores
                scores.append((score, doc))

        # Sort by score (highest first) and return top k
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Convert to distance format (lower is better) for consistency with VectorIndex
        return [(doc, math.exp(-0.1 * score)) for score, doc in scores[:k]]

    def _compute_bm25_score(self, query_tokens: List[str], doc_index: int) -> float:
        """Compute BM25 score for a document given query tokens"""
        score = 0.0
        doc_term_counts = self._corpus_term_counts[doc_index]
        doc_length = self._doc_len[doc_index]

        for token in query_tokens:
            if token not in self._idf:
                continue

            idf = self._idf[token]
            term_freq = doc_term_counts.get(token, 0)

            # BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (|d| / avgdl)))
            numerator = idf * term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self._avg_doc_len))
            score += numerator / (denominator + 1e-9)  # Small epsilon to avoid division by zero

        return score


class RAGPipelineV2:
    """Hybrid local retrieval with optional contextual indexing and reranking."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = select_device()
        print(f"[RAG V2] Using compute device: {self.device}")

        self.contextualizer = self.config.get("contextualizer")
        if self.config.get("use_contextual", False) and self.contextualizer is None:
            self.contextualizer = AnthropicContextualizer(
                model=str(self.config.get("contextual_model", "claude-haiku-4-5")),
                max_tokens=int(self.config.get("contextual_max_tokens", 128)),
                timeout=float(self.config.get("contextual_timeout", 30)),
            )

        # Embeddings come from the user's `rag setup` choice (or RAG_MCP_* env).
        # None means exact-word (BM25) search only.
        embed_provider = str(self.config.get("embed_provider") or "none").lower()
        self.embedding_fn = build_embedder(self.config, device=self.device)
        if self.embedding_fn is None:
            print("[RAG V2] No embedding model configured: exact-word (BM25) search only. Run `rag setup` to add meaning-based search.")
        else:
            print(f"[RAG V2] Using {embed_provider} embeddings with model: {self.embedding_fn.model_name}")

        # Determine vector database backend (LanceDB by default, or Qdrant)
        self.backend_type = self.config.get("backend", os.environ.get("RAG_MCP_BACKEND", "lancedb")).lower()
        print(f"[RAG V2] Using backend: {self.backend_type}")

        # One probe embedding tells the real vector size for any provider/model.
        vector_size = probe_dimensions(self.embedding_fn) if self.embedding_fn is not None else 0

        # Create unique collection name from persist_dir to avoid conflicts
        persist_dir_name = os.path.basename(self.config.get("persist_dir"))
        collection_name = f"rag_v2_{persist_dir_name}" if persist_dir_name != "rag_db_v2" else "rag_v2_default"

        if self.backend_type == "lancedb":
            lance_path = os.path.join(self.config.get("persist_dir"), "lancedb_storage")
            self.vector_db = LanceDBBackend(
                table_name=collection_name,
                embedding_fn=self.embedding_fn,
                vector_size=vector_size,
                distance=self.config.get("distance_metric", "cosine"),
                persist_path=lance_path,
            )
            self.bm25_index = None
        else:
            if self.embedding_fn is None:
                raise ValueError("The Qdrant backend needs an embedding model. Run `rag setup` or use the LanceDB backend.")
            from .qdrant_backend import QdrantVectorDB

            qdrant_path = os.path.join(self.config.get("persist_dir"), "qdrant_storage")
            self.vector_db = QdrantVectorDB(
                collection_name=collection_name,
                embedding_fn=self.embedding_fn,
                vector_size=vector_size,
                distance=self.config.get("distance_metric", "cosine"),
                persist_path=qdrant_path
            )
            # Keep BM25 (still valuable for lexical search)
            self.bm25_index = BM25Index(
                k1=self.config.get("bm25_k1", 1.5),
                b=self.config.get("bm25_b", 0.75)
            )

        # Add CrossEncoder / LLM for fast reranking
        reranker_model = self.config.get('reranker_model') or ""
        reranker_setting = str(self.config.get("reranker_type", "auto")).lower()
        self.reranker_type = "cross-encoder"
        self.reranker = None
        if reranker_setting == "auto":
            try:
                import sentence_transformers  # noqa: F401
                reranker_setting = "cross-encoder"
            except ImportError:
                reranker_setting = "disabled"
        if reranker_setting in {"disabled", "none", "off"}:
            print("[RAG V2] Reranker off.")
        elif reranker_setting == "fastembed":
            from .embeddings import FastEmbedReranker
            self.reranker = FastEmbedReranker(reranker_model or "Xenova/ms-marco-MiniLM-L-6-v2")
            print(f"[RAG V2] Using local ONNX reranker: {self.reranker.model_name}")
        else:
            reranker_model = reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            if "qwen3-reranker" in reranker_model.lower():
                self.reranker_type = "qwen3"
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                print(f"[RAG V2] Loading Qwen3 Reranker on GPU from local cache: {reranker_model}")
                self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                    reranker_model,
                    padding_side='left',
                    local_files_only=True
                )
                self.reranker_model = AutoModelForCausalLM.from_pretrained(
                    reranker_model,
                    torch_dtype=torch.float16,
                    device_map="cuda",
                    local_files_only=True
                ).eval()

            else:
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError:
                    print("[RAG V2] Reranker skipped: sentence-transformers is not installed (full extra). Results keep their fusion order.")
                else:
                    self.reranker = CrossEncoder(
                        reranker_model,
                        device=self.device
                    )
                    print(f"[RAG V2] Using CrossEncoder reranking with model: {reranker_model}")

        # Load or create database
        self._load_or_create_database()

    _extract_text_from_pdf = parsing._extract_text_from_pdf
    _extract_text_with_unstructured = parsing._extract_text_with_unstructured
    _extract_content = parsing._extract_content
    _extract_text_from_notebook = parsing._extract_text_from_notebook
    _extract_text_from_json = parsing._extract_text_from_json

    def _locate_chunk(self, full_text: str, content: str, ext: str, search_from: int) -> tuple:
        """Recover a human-readable location for a chunk: a page number for PDFs
        (pages joined with \\f during extraction), a cell number for notebooks
        (cells joined with \\f), or a line range for everything else, where
        newlines survive extraction untouched. Returns (location_string, next
        search_from) so callers can scan the document once, left to right."""
        start = full_text.find(content, search_from)
        if start == -1:
            start = full_text.find(content)
        if start == -1:
            return ("", search_from)

        if ext == ".pdf":
            page = full_text.count("\f", 0, start) + 1
            location = f"p.{page}"
        elif ext == ".ipynb":
            cell = full_text.count("\f", 0, start) + 1
            location = f"cell {cell}"
        else:
            line_start = full_text.count("\n", 0, start) + 1
            line_end = line_start + content.count("\n")
            location = f"L{line_start}" if line_end == line_start else f"L{line_start}-L{line_end}"
        return (location, start + 1)

    def _add_context_to_chunk(self, chunk: str, document_context: str) -> str:
        if not self.config.get("use_contextual", False):
            return chunk

        try:
            context = self.contextualizer(chunk, document_context).strip()
        except Exception as error:
            print(
                f"[RAG V2] Contextual retrieval failed; keeping the original chunk: {error}",
                file=sys.stderr,
            )
            return chunk
        return f"{context}\n{chunk}" if context else chunk

    def _load_or_create_database(self):
        """Load or create the persisted database."""
        persist_dir = self.config.get("persist_dir")

        if not os.path.exists(persist_dir) or self.config.get("force_reindex", False):
            print(f"[RAG V2] Creating new database at {persist_dir}...")
            if self.config.get("force_reindex", False):
                try:
                    self.vector_db.clear()
                except Exception as e:
                    print(f"[RAG V2] Warning: Failed to clear Qdrant collection: {e}")
            self._index_documents()
        else:
            print(f"[RAG V2] Loading existing database from {persist_dir}...")
            self._load_existing_database()

    def _index_documents(self):
        """Extract, chunk, optionally contextualize, and persist documents."""
        import concurrent.futures
        all_docs = []
        document_contents = {}

        # Collect all file paths first
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

        def process_file(file_path):
            content = self._extract_content(file_path)
            if content:
                return file_path, content
            return None

        # First pass: extract all document content using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_file, files_to_process)
            for result in results:
                if result:
                    file_path, content = result
                    document_contents[file_path] = content

        # Second pass: chunk and optionally add context
        for file_path, full_content in document_contents.items():
            print(f"[RAG V2] Processing {os.path.basename(file_path)}...")

            chunks = chunk_document(
                full_content,
                file_path=file_path,
                chunk_size=self.config.get("chunk_size"),
                chunk_overlap=self.config.get("chunk_overlap")
            )

            ext = os.path.splitext(file_path)[1].lower()
            raw_chunks = [item["content"].replace("\f", " ") for item in chunks]
            search_from = 0
            for i, item in enumerate(chunks):
                chunk = item["content"]
                if chunk.strip():
                    location, search_from = self._locate_chunk(full_content, chunk, ext, search_from)
                    clean_chunk = raw_chunks[i]

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
                    contextualized_chunk = self._add_context_to_chunk(clean_chunk, document_context)

                    doc = {
                        "content": contextualized_chunk,
                        "original_content": clean_chunk,  # Keep original for comparison
                        "source": file_path,
                        "file_path": file_path,
                        "location": location,
                        "start_line": int(location[1:].split("-")[0]) if location.startswith("L") else 0,
                        "end_line": int(location.rsplit("L", 1)[1]) if location.startswith("L") else 0,
                        "page_number": int(location[2:]) if location.startswith("p.") else 0,
                        "heading": item["heading"],
                        "chunk_id": i,
                        "id": f"{os.path.realpath(file_path)}::chunk:{i}"
                    }
                    all_docs.append(doc)

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


    def _save_database(self, docs: List[Dict]):
        """Save the database (simplified for now)"""
        persist_dir = self.config.get("persist_dir")
        os.makedirs(persist_dir, exist_ok=True)
        
        # Save document metadata
        with open(os.path.join(persist_dir, "docs.json"), 'w') as f:
            json.dump(docs, f, indent=2)

    def _load_existing_database(self):
        """Load existing database"""
        persist_dir = self.config.get("persist_dir")
        docs_file = os.path.join(persist_dir, "docs.json")

        db_count = self.vector_db.count()

        if db_count > 0 and os.path.exists(docs_file):
            with open(docs_file, 'r') as f:
                docs = json.load(f)

            if self.bm25_index is not None:
                print(f"[RAG V2] Loading {len(docs)} chunks and rebuilding BM25...")
                for doc in docs:
                    self.bm25_index.add_document(doc)
                self.bm25_index._build_index()

            print(f"[RAG V2] Loaded {len(docs)} chunks from existing database ({self.backend_type}: {db_count})")
        else:
            # Database directory exists but empty - trigger indexing
            print(f"[RAG V2] Database empty. Triggering indexing...")
            self._index_documents()

    def _rrf_fusion(self, vector_results: List[Tuple], bm25_results: List[Tuple], k: int = 60) -> List[Dict]:
        """
        STEP 5: HYBRID SEARCH FUSION (Unified Architecture)
        
        This unifies our V1 (Weighted Ensemble) and V2 (Unweighted RRF) architectures.
        It uses Reciprocal Rank Fusion (RRF) to merge keyword (BM25) and semantic (Vector) search.
        
        If 'ensemble_weights' is provided in the config (e.g. [0.5, 0.5]), it performs a 
        Weighted RRF, exactly matching LangChain's EnsembleRetriever math (V1).
        If no weights are provided, it falls back to standard Unweighted RRF (V2).
        
        RRF Formula: score = weight * (1 / (k + rank))
        """
        doc_scores = {}
        
        # Check if the user wants tunable weights (V1 style)
        weights = self.config.get("ensemble_weights", [1.0, 1.0])
        
        # Retain the first record and the last rank per source for repeated IDs.
        for rank_key, results in (("vector_rank", vector_results), ("bm25_rank", bm25_results)):
            for rank, (doc, distance) in enumerate(results, 1):
                doc_id = doc.get("id", id(doc))
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"doc": doc, "vector_rank": float('inf'), "bm25_rank": float('inf')}
                doc_scores[doc_id][rank_key] = rank

        # Calculate Unified RRF scores
        rrf_results = []
        for doc_id, data in doc_scores.items():
            vector_score = weights[0] * (1.0 / (k + data["vector_rank"])) if data["vector_rank"] != float('inf') else 0
            bm25_score = weights[1] * (1.0 / (k + data["bm25_rank"])) if data["bm25_rank"] != float('inf') else 0
            
            rrf_score = vector_score + bm25_score
            if rrf_score > 0:
                rrf_results.append((data["doc"], rrf_score))

        # Sort by RRF score (higher is better)
        rrf_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in rrf_results]

    def _llm_rerank(self, docs: List[Dict], query: str, k: int) -> List[Dict]:
        """Preserve fused order for the unsupported legacy LLM setting."""
        print("[RAG Core] Executing LLM-based Reranking...")
        return docs[:k]


    def _crossencoder_rerank(self, docs: List[Dict], query: str, k: int) -> List[Dict]:
        """
        STEP 6: CROSSENCODER RERANKING

        Use CrossEncoder to rerank documents by relevance.
        Fast, GPU-based, task-specialized for passage ranking.
        """
        if len(docs) <= k:
            return docs

        try:
            if self.reranker_type == "qwen3":
                import torch
                token_false_id = self.reranker_tokenizer.convert_tokens_to_ids("no")
                token_true_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")
                prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
                suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                prefix_tokens = self.reranker_tokenizer.encode(prefix, add_special_tokens=False)
                suffix_tokens = self.reranker_tokenizer.encode(suffix, add_special_tokens=False)
                instruction = 'Given a web search query, retrieve relevant passages that answer the query'
                
                scores = []
                if not docs:
                    return []

                if self.reranker_tokenizer.pad_token_id is None:
                    self.reranker_tokenizer.pad_token_id = self.reranker_tokenizer.eos_token_id

                input_ids_list = []
                max_len = 0
                for doc in docs:
                    text = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc['content'][:1000]}"
                    inputs = self.reranker_tokenizer(text, return_attention_mask=False, add_special_tokens=False)
                    input_ids = prefix_tokens + inputs['input_ids'] + suffix_tokens
                    input_ids_list.append(input_ids)
                    if len(input_ids) > max_len:
                        max_len = len(input_ids)

                pad_token_id = self.reranker_tokenizer.pad_token_id
                padded_input_ids = []
                attention_masks = []

                for ids in input_ids_list:
                    pad_len = max_len - len(ids)
                    # Left padding for decoder-only models
                    padded_input_ids.append([pad_token_id] * pad_len + ids)
                    attention_masks.append([0] * pad_len + [1] * len(ids))

                inputs_tensor = torch.tensor(padded_input_ids).to(self.device)
                attention_mask_tensor = torch.tensor(attention_masks).to(self.device)

                with torch.no_grad():
                    outputs = self.reranker_model(inputs_tensor, attention_mask=attention_mask_tensor)
                    batch_scores = outputs.logits[:, -1, :]
                    true_vector = batch_scores[:, token_true_id]
                    false_vector = batch_scores[:, token_false_id]
                    batch_scores_stack = torch.stack([false_vector, true_vector], dim=1)
                    batch_scores_log = torch.nn.functional.log_softmax(batch_scores_stack, dim=1)
                    scores = batch_scores_log[:, 1].exp().tolist()
                    
                # Aggressively free memory
                del inputs_tensor
                del attention_mask_tensor
                del outputs
                del batch_scores
                del true_vector
                del false_vector
                del batch_scores_stack
                del batch_scores_log
                torch.cuda.empty_cache()
            elif hasattr(self, 'reranker') and self.reranker:
                pairs = [[query, doc['content'][:512]] for doc in docs]  # Limit to 512 chars for speed
                scores = self.reranker.predict(pairs)
            else:
                scores = [1.0] * len(docs)

            # Sort documents by score (higher is better)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

            return [doc for doc, score in ranked[:k]]

        except Exception as e:
            print(f"[RAG Core] CrossEncoder reranking failed: {e}, falling back to original order")
            return docs[:k]

    def search(self, query: str) -> str:
        """Return ranked passages from the configured local retrieval pipeline."""
        try:
            top_k = self.config.get("top_k", 20)
            rerank_top_k = self.config.get("rerank_top_k", 3)
            
            print(f"[RAG V2] Searching with query: '{query[:50]}...' (backend: {self.backend_type})")
            
            if self.backend_type == "lancedb":
                vector_results = []
                if self.embedding_fn is not None:
                    # STEP 3: LANCEDB VECTOR SEARCH
                    print("[RAG V2] Step 3: LanceDB Vector search...")
                    vector_results_raw = self.vector_db.search_vector(query, top_k=top_k)
                    vector_results = [(doc, doc.get("score", 0)) for doc in vector_results_raw]
                else:
                    print("[RAG V2] Step 3: skipped (no embedding model; exact-word search only)")

                # STEP 4: LANCEDB TANTIVY FTS SEARCH
                print("[RAG V2] Step 4: LanceDB Tantivy FTS search...")
                bm25_results_raw = self.vector_db.search_fts(query, top_k=top_k)
                bm25_results = [(doc, doc.get("score", 0)) for doc in bm25_results_raw]
            else:
                # STEP 3: QDRANT VECTOR SEARCH
                print("[RAG V2] Step 3: Vector search...")
                vector_results_qdrant = self.vector_db.search(query, top_k=top_k)
                # Convert to tuple format for RRF
                vector_results = [(doc, doc.get("score", 0)) for doc in vector_results_qdrant]

                # STEP 4: BM25 SEARCH
                print("[RAG V2] Step 4: BM25 search...")
                bm25_results = self.bm25_index.search(query, k=top_k)
            
            # STEP 5: HYBRID FUSION with RRF
            print("[RAG V2] Step 5: RRF fusion...")
            fused_docs = self._rrf_fusion(vector_results, bm25_results, k=self.config.get("rrf_k", 60))
            
            if not fused_docs:
                return "No relevant information found."
            # Step 4: Reranking (Configurable via reranker_type)
            reranker_type = str(self.config.get("reranker_type", "auto")).lower()
            
            if reranker_type in {"disabled", "none", "off"}:
                final_docs = fused_docs[:rerank_top_k]
            elif self.reranker_type != "qwen3" and getattr(self, "reranker", None) is None:
                # No reranker model is loaded (light install): keep the fusion order.
                final_docs = fused_docs[:rerank_top_k]
            elif reranker_type == "llm":
                print(f"[RAG Core] Reranking {len(fused_docs)} initial results using LLM Reranker...")
                final_docs = self._llm_rerank(fused_docs, query, k=rerank_top_k)
            else:
                print(f"[RAG Core] Reranking {len(fused_docs)} initial results using CrossEncoder Reranker...")
                final_docs = self._crossencoder_rerank(fused_docs, query, k=rerank_top_k)
            sys.stderr.write(f"[RAG V2] Returning {len(final_docs)} unique results\n")
            return final_docs
            
        except Exception as e:
            print(f"[RAG V2] Search error: {e}")
            return []
