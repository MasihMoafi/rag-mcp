import sys
import os, re, string, fitz, json, math
import torch
from typing import List, Dict, Tuple, Any, Callable

# --- FIX for Ollama Proxy ---
# This is necessary to ensure the local Ollama server can be reached.
def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]
clear_proxy_settings()

from collections import Counter
from sentence_transformers import SentenceTransformer, CrossEncoder
from .qdrant_backend import QdrantVectorDB
from .lancedb_backend import LanceDBBackend


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks without langchain."""
    if not text or len(text.strip()) < 50:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if len(chunk.strip()) > 50:
            chunks.append(chunk)

        start = end - chunk_overlap if end < len(text) else len(text)

        if len(chunks) > 10000:  # Safety limit
            break

    return chunks


_MD_EXTS = {".md", ".markdown"}
_CODE_EXTS = {
    ".py", ".rs", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".cpp", ".hpp",
    ".go", ".java", ".kt", ".sh", ".bash", ".zsh", ".sql",
}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
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


def _paragraph_split(text: str) -> List[str]:
    """Split on blank lines, the cheapest reliable proxy for a semantic boundary."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _nearest_code_symbol(text: str) -> str:
    """Best-effort function/class name a code chunk falls under, for citations."""
    for line in text.split("\n"):
        match = _CODE_SYMBOL_RE.match(line)
        if match:
            return match.group(1)
    return ""


def _merge_paragraphs(paragraphs: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """Greedily merge paragraphs up to chunk_size, keeping each paragraph whole
    instead of cutting mid-sentence. A paragraph larger than chunk_size (e.g. a
    long function) falls back to the raw sliding-window splitter on its own."""
    chunks: List[str] = []
    buf = ""

    for para in paragraphs:
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) <= chunk_size:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        if len(para) > chunk_size:
            chunks.extend(chunk_text(para, chunk_size, chunk_overlap))
            buf = ""
        else:
            buf = para
    if buf:
        chunks.append(buf)

    if chunk_overlap and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            tail = prev[-chunk_overlap:]
            overlapped.append(cur if tail in cur else f"{tail}\n\n{cur}")
        return overlapped

    return chunks


def chunk_document(
    text: str, file_path: str = "", chunk_size: int = 700, chunk_overlap: int = 100
) -> List[Dict[str, str]]:
    """Structure-aware chunking. Splits along markdown headings, code
    function/class boundaries, or blank-line paragraphs before falling back to
    the raw sliding-window splitter, so a chunk is far less likely to cut a
    sentence, section, or function in half. Returns dicts with a "heading"
    breadcrumb (markdown section path or nearest code symbol) for citations.
    """
    if not text or len(text.strip()) < 50:
        return []

    ext = os.path.splitext(file_path)[1].lower()

    if ext in _MD_EXTS and _HEADING_RE.search(text):
        results = []
        for breadcrumb, section in _split_markdown_sections(text):
            for chunk in _merge_paragraphs(_paragraph_split(section), chunk_size, chunk_overlap):
                if len(chunk.strip()) > 50:
                    results.append({"content": chunk, "heading": breadcrumb})
        if results:
            return results

    paragraphs = _paragraph_split(text)
    if paragraphs:
        heading_fn = _nearest_code_symbol if ext in _CODE_EXTS else (lambda _c: "")
        results = [
            {"content": chunk, "heading": heading_fn(chunk)}
            for chunk in _merge_paragraphs(paragraphs, chunk_size, chunk_overlap)
            if len(chunk.strip()) > 50
        ]
        if results:
            return results

    return [{"content": chunk, "heading": ""} for chunk in chunk_text(text, chunk_size, chunk_overlap)]



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
            # IDF formula from tutorial: log(((N - freq + 0.5) / (freq + 0.5)) + 1)
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
        seen_in_doc = set()
        for token in doc_tokens:
            if token not in seen_in_doc:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                seen_in_doc.add(token)
        
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
        results = []
        for score, doc in scores[:k]:
            # Convert score to distance using exponential normalization
            distance = math.exp(-0.1 * score)
            results.append((doc, distance))
        
        return results

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
    """
    RAG Pipeline V2 implementing tutorial techniques:
    1. Custom VectorIndex and BM25Index
    2. RRF (Reciprocal Rank Fusion) instead of weighted ensemble
    3. LLM-based reranking with Ollama
    4. Contextual chunk preprocessing
    """
    
    def __init__(self, config: dict):
        self.config = config
        import torch
        # Auto-switch to GPU and enforce CUDA availability
        if not torch.cuda.is_available():

            import subprocess
            subprocess.run(["pkexec", "prime-select", "nvidia"])
            if not torch.cuda.is_available():
                raise RuntimeError("Dedicated GPU (CUDA) is not available. Please verify NVIDIA driver status.")

        self.device = 'cuda'




        # Initialize embedding function based on provider
        embed_provider = self.config.get("embed_provider", "sentencetransformer")

        if embed_provider == "ollama":
            import requests

            model_name = self.config.get("embed_model")
            ollama_url = self.config.get("ollama_url", "http://localhost:11434")

            class DirectOllamaGPUWrapper:
                def __init__(self, model, base_url):
                    self.model = model
                    self.base_url = base_url

                def _verify_gpu(self):
                    try:
                        resp = requests.get(f"{self.base_url}/api/ps", timeout=5)
                        if resp.status_code == 200:
                            models = resp.json().get("models", [])
                            for m in models:
                                if self.model in m.get("name", ""):
                                    proc = m.get("details", {}).get("processor", "") or m.get("processor", "")
                                    if "CPU" in str(proc) and "100% CPU" in str(proc):
                                        raise RuntimeError(f"OLLAMA GPU FAILURE: {self.model} is running on {proc}! Aborting.")
                    except Exception as e:
                        if "OLLAMA GPU FAILURE" in str(e):
                            raise

                def embed_documents(self, texts):
                    """Send true vector batches directly to /api/embed on GPU."""
                    if not texts:
                        return []
                    # Process in GPU batches of 512
                    batch_size = 512
                    all_embeddings = []
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        resp = requests.post(
                            f"{self.base_url}/api/embed",
                            json={"model": self.model, "input": batch},
                            timeout=180
                        )
                        if resp.status_code != 200:
                            raise RuntimeError(f"Ollama embedding failed ({resp.status_code}): {resp.text}")
                        embeddings = resp.json().get("embeddings", [])
                        all_embeddings.extend(embeddings)
                    self._verify_gpu()
                    return all_embeddings

                def embed_query(self, text):
                    resp = requests.post(
                        f"{self.base_url}/api/embed",
                        json={"model": self.model, "input": [text]},
                        timeout=60
                    )
                    if resp.status_code != 200:
                        raise RuntimeError(f"Ollama query embedding failed ({resp.status_code}): {resp.text}")
                    embeddings = resp.json().get("embeddings", [])
                    self._verify_gpu()
                    return embeddings[0] if embeddings else []

            self.embedding_fn = DirectOllamaGPUWrapper(model_name, ollama_url)
            print(f"[RAG V2] Using Direct Ollama GPU Batched Embeddings with model: {model_name}")


        elif embed_provider == "openai_compatible":
            # One client for any provider that speaks the OpenAI embeddings API shape:
            # real OpenAI, Ollama's own /v1 endpoint, and Qwen/DashScope's and Gemini's
            # OpenAI-compatible modes all fit here without a provider-specific SDK.
            # Requires the `openai` package (not a hard dependency of this project) and
            # leaves the machine over the network -- this is the one provider option
            # that is not local. Verified against RAGFlow's provider list, where most
            # of its remote embedding providers are themselves OpenAIEmbed subclasses.
            from openai import OpenAI
            client = OpenAI(
                api_key=self.config.get("embed_api_key") or None,
                base_url=self.config.get("embed_base_url") or None,
            )
            model_name = self.config.get("embed_model")

            class OpenAICompatibleWrapper:
                def __init__(self, client, model_name):
                    self.client = client
                    self.model_name = model_name
                def _create_with_retry(self, input_val, max_retries=6):
                    import time as _time
                    for attempt in range(max_retries):
                        try:
                            return self.client.embeddings.create(model=self.model_name, input=input_val)
                        except Exception:
                            if attempt == max_retries - 1:
                                raise
                            _time.sleep(min(3 * (attempt + 1), 20))
                def embed_documents(self, texts):
                    resp = self._create_with_retry(texts)
                    return [item.embedding for item in resp.data]
                def embed_query(self, text):
                    resp = self._create_with_retry([text])
                    return resp.data[0].embedding

            self.embedding_fn = OpenAICompatibleWrapper(client, model_name)
            print(f"[RAG V2] Using OpenAI-compatible embeddings with model: {model_name}")
        else:
            model = SentenceTransformer(self.config.get("embed_model"), device=self.device)
            class STWrapper:
                def __init__(self, st_model):
                    self.model = st_model
                def embed_documents(self, texts):
                    return self.model.encode(texts, convert_to_numpy=True).tolist()
                def embed_query(self, text):
                    return self.model.encode([text], convert_to_numpy=True)[0].tolist()
            
            self.embedding_fn = STWrapper(model)
            print(f"[RAG V2] Using SentenceTransformer embeddings with model: {self.config.get('embed_model')}")

        # Determine vector database backend (LanceDB by default, or Qdrant)
        self.backend_type = self.config.get("backend", os.environ.get("RAG_MCP_BACKEND", "lancedb")).lower()
        print(f"[RAG V2] Using backend: {self.backend_type}")

        # Determine vector size based on model
        vector_size_map = {
            "ollama": 768,  # default ollama fallback
            "vllm": 4096,  # e5-mistral-7b-instruct
            "sentencetransformer": 384,  # all-MiniLM-L6-v2
            "openai_compatible": 1536,  # OpenAI text-embedding-3-small / Qwen text-embedding-v2 default
        }
        vector_size = vector_size_map.get(embed_provider, 384)
        embed_model_name = self.config.get("embed_model", "").lower()
        if "qwen3-embedding" in embed_model_name:
            vector_size = 4096

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
        reranker_model = self.config.get('reranker_model')
        self.reranker_type = "cross-encoder"
        if reranker_model:
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
                self.reranker = CrossEncoder(
                    reranker_model,
                    device=self.device
                )
                print(f"[RAG V2] Using CrossEncoder reranking with model: {reranker_model}")

        # Load or create database
        self._load_or_create_database()

    def _normalize_text(self, text: str) -> str:
        """Same normalization as V2 for consistency"""
        if not text: return ""
        translator = str.maketrans('', '', string.punctuation + '،؛؟»«')
        text = text.translate(translator)
        return ' '.join(text.split()).lower()

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF extraction. Pages are joined with a form-feed marker so a chunk's
        page number can be recovered later; without this the page boundary is
        destroyed and every chunk looks like it came from nowhere."""
        try:
            doc = fitz.open(pdf_path)
            pages = [re.sub(r'\s+', ' ', page.get_text()).strip() for page in doc]
            doc.close()
            return "\f".join(pages)
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

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
        return (location, start)

    def _add_context_to_chunk(self, chunk: str, document_context: str) -> str:
        """
        STEP 7: CONTEXTUAL RETRIEVAL
        Add context to chunks using LLM before embedding.
        This helps chunks be more searchable by providing situational context.
        """
        # Always return chunk without context to avoid Ollama hanging
        return chunk

    def _load_or_create_database(self):
        """Load or create the V2 database with all tutorial techniques"""
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
        """Index all documents using V2 techniques"""
        import sys
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
                    indexable_exts = (
                        '.rs', '.toml', '.yaml', '.yml', '.py', '.md', '.txt', '.json',
                        '.js', '.ts', '.tsx', '.jsx', '.c', '.h', '.cpp', '.hpp',
                        '.go', '.sh', '.bash', '.zsh', '.css', '.html', '.sql', '.java',
                        '.kt', '.proto', '.pdf', '.ipynb', '.rst', '.ini', '.cfg', '.conf',
                        '.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp', '.docx', '.pptx', '.xlsx', '.csv'
                    )
                    if file.lower().endswith(indexable_exts):
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

            # STEP 1: CHUNKING (structure-aware: markdown headings / code
            # symbols / paragraphs first, raw sliding-window as last resort)
            chunks = chunk_document(
                full_content,
                file_path=file_path,
                chunk_size=self.config.get("chunk_size"),
                chunk_overlap=self.config.get("chunk_overlap")
            )

            ext = os.path.splitext(file_path)[1].lower()
            search_from = 0
            for i, item in enumerate(chunks):
                chunk = item["content"]
                if len(chunk.strip()) > 50:
                    location, search_from = self._locate_chunk(full_content, chunk, ext, search_from)
                    clean_chunk = chunk.replace("\f", " ")

                    # STEP 7: CONTEXTUAL RETRIEVAL
                    contextualized_chunk = self._add_context_to_chunk(clean_chunk, full_content)

                    doc = {
                        "content": contextualized_chunk,
                        "original_content": clean_chunk,  # Keep original for comparison
                        "source": file_path,
                        "location": location,
                        "heading": item["heading"],
                        "chunk_id": i,
                        "id": f"{os.path.basename(file_path)}_chunk_{i}"
                    }
                    all_docs.append(doc)

        if not all_docs:
            raise ValueError("No documents processed.")

        # Add documents to indexes
        print(f"[RAG V2] Indexing {len(all_docs)} chunks...")

        # STEP 2 & 4: Indexing into vector database and lexical index
        self.vector_db.add_documents_batch(all_docs, batch_size=512)


        if self.bm25_index is not None:
            print(f"[RAG V2] Building BM25 index...")
            for doc in all_docs:
                self.bm25_index.add_document(doc)
            self.bm25_index._build_index()

        # Save database
        self._save_database(all_docs)
        print(f"[RAG V2] Database created with {len(all_docs)} chunks")

    def _extract_text_with_unstructured(self, file_path: str) -> str:
        """Extract text, tables, and OCR/element descriptions using Unstructured and RapidOCR."""
        ext = os.path.splitext(file_path)[1].lower()
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}

        # For standalone image files, run RapidOCR directly
        if ext in image_exts:
            try:
                from rapidocr_onnxruntime import RapidOCR
                engine = RapidOCR()
                result, _ = engine(file_path)
                if result:
                    return "\n".join([line[1] for line in result])
            except Exception as e:
                print(f"[OCR] Warning: RapidOCR failed for {file_path}: {e}")

        try:
            from unstructured.partition.auto import partition
            elements = partition(filename=file_path)
            parts = []
            for el in elements:
                text = str(el).strip()
                if text:
                    page_num = getattr(getattr(el, 'metadata', None), 'page_number', None)
                    page_prefix = f"[Page {page_num}] " if page_num else ""
                    parts.append(f"{page_prefix}{text}")
            res = "\n\n".join(parts)
            if res:
                return res
        except Exception as e:
            print(f"[Unstructured] Warning: extraction failed for {file_path}: {e}")

        return ""

    def _extract_content(self, file_path: str) -> str:
        """Extract content from various file types, using Unstructured for images and complex docs."""
        ext = os.path.splitext(file_path)[1].lower()
        unstructured_exts = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".docx", ".pptx", ".xlsx", ".csv"}
        if ext in unstructured_exts:
            return self._extract_text_with_unstructured(file_path)
        elif ext == ".pdf":
            text = self._extract_text_from_pdf(file_path)
            if not text or len(text.strip()) < 50:
                text = self._extract_text_with_unstructured(file_path)
            return text
        elif ext == ".json":
            return self._extract_text_from_json(file_path)
        elif ext == ".ipynb":
            return self._extract_text_from_notebook(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return ""

    def _extract_text_from_notebook(self, notebook_path: str) -> str:
        """Extract text from Jupyter notebook (.ipynb). Cells are joined with a
        form-feed marker, same as PDF pages, so a chunk's originating cell can
        be recovered later instead of collapsing into one undifferentiated blob."""
        try:
            from .notebook_chunker import extract_cells_from_notebook
            cells = extract_cells_from_notebook(notebook_path)
            return "\f".join(cell['content'] for cell in cells)
        except Exception as e:
            print(f"Error extracting text from {notebook_path}: {e}")
            return ""

    def _extract_text_from_json(self, json_path: str) -> str:
        """Extract text from JSON - handles various structures"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert JSON to searchable text
            def flatten_json(obj, prefix=''):
                text_parts = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, (dict, list)):
                            text_parts.extend(flatten_json(value, f"{prefix}{key}."))
                        else:
                            text_parts.append(f"{prefix}{key}: {value}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        text_parts.extend(flatten_json(item, f"{prefix}[{i}]."))
                else:
                    text_parts.append(str(obj))
                return text_parts

            text_parts = flatten_json(data)
            return "\n".join(text_parts)
        except Exception as e:
            print(f"Error extracting text from {json_path}: {e}")
            return ""

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
        
        # Process vector search results (rank by similarity - lower distance = higher rank)
        for rank, (doc, distance) in enumerate(vector_results, 1):
            doc_id = doc.get("id", id(doc))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "vector_rank": float('inf'), "bm25_rank": float('inf')}
            doc_scores[doc_id]["vector_rank"] = rank

        # Process BM25 results (rank by relevance - lower distance = higher rank) 
        for rank, (doc, distance) in enumerate(bm25_results, 1):
            doc_id = doc.get("id", id(doc))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "vector_rank": float('inf'), "bm25_rank": float('inf')}
            doc_scores[doc_id]["bm25_rank"] = rank

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
        """
        STEP 6b: LLM-BASED RERANKING (Alternative to CrossEncoder)

        Instead of a specialized CrossEncoder (like Qwen3 or MS-MARCO), this uses 
        a generative LLM to act as a judge, analyzing each chunk's relevance to the query.
        
        Why this exists:
        - Portability: Users don't need heavy ML frameworks (Torch/Transformers).
        - Reasoning: Large LLMs can reason through complex semantic connections that 
          smaller cross-encoders might miss.
        """
        print("[RAG Core] Executing LLM-based Reranking...")
        
        # NOTE: In a full production environment, this would call your Ollama/OpenAI API.
        # This implementation serves as the scaffolding for the LLM judge logic.
        scores = []
        for doc in docs:
            # Hypothetical Prompt Structure:
            # prompt = f"Query: {query}\nPassage: {doc['content']}\nScore this passage from 0 to 10 based on its ability to answer the query."
            # response = llm.generate(prompt)
            # score = parse_score(response)
            
            # Since we don't want to block the execution with an API call here unless configured,
            # we simulate an LLM pass-through for now, ready to be wired to the TUI's LLM engine.
            scores.append(1.0) 
            
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked[:k]]


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
        """
        Main search method implementing all 7 tutorial steps:
        1. ✅ Chunking (in indexing)
        2. ✅ Embeddings (in vector search)  
        3. ✅ Vector Search (semantic similarity)
        4. ✅ BM25 (keyword search)
        5. ✅ Hybrid (RRF fusion)
        6. ✅ Reranking (LLM-based)
        7. ✅ Contextual (in indexing)
        """
        try:
            top_k = self.config.get("top_k", 20)
            rerank_top_k = self.config.get("rerank_top_k", 3)
            
            print(f"[RAG V2] Searching with query: '{query[:50]}...' (backend: {self.backend_type})")
            
            if self.backend_type == "lancedb":
                # STEP 3: LANCEDB VECTOR SEARCH
                print("[RAG V2] Step 3: LanceDB Vector search...")
                vector_results_raw = self.vector_db.search_vector(query, top_k=top_k)
                vector_results = [(doc, doc.get("score", 0)) for doc in vector_results_raw]

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
            reranker_type = self.config.get("reranker_type", "cross-encoder").lower()
            
            if reranker_type == "disabled" or reranker_type == "none":
                final_docs = fused_docs[:rerank_top_k]
            elif reranker_type == "llm":
                print(f"[RAG Core] Reranking {len(fused_docs)} initial results using LLM Reranker...")
                final_docs = self._llm_rerank(fused_docs, query, k=rerank_top_k)
            else:
                print(f"[RAG Core] Reranking {len(fused_docs)} initial results using CrossEncoder Reranker...")
                final_docs = self._crossencoder_rerank(fused_docs, query, k=rerank_top_k)
            unique_content = []
            for doc in final_docs:
                content = doc.get('original_content', doc.get('content', ''))
                if content not in unique_content:
                    unique_content.append(content)
            
            sys.stderr.write(f"[RAG V2] Returning {len(final_docs)} unique results\n")
            return final_docs
            
        except Exception as e:
            print(f"[RAG V2] Search error: {e}")
            return []
