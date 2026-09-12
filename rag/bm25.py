"""In-memory lexical index for the Qdrant retrieval branch."""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple


class BM25Index:
    """
    Track corpus statistics and return ranked documents as distance scores.
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
