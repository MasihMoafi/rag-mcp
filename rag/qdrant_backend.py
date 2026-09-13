"""
Qdrant Vector Database Backend for RAG V2

Replaces custom VectorIndex with production-ready Qdrant.
Key improvements:
- Batch embedding (10x+ faster indexing)
- Persistent storage
- Production-grade vector search
"""

import sys

# Central proxy manager — strip SOCKS, keep HTTP(S) for cloud
from utils.proxy import strip_socks as _strip_socks
_strip_socks()

from typing import List, Dict, Any, Callable
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from .backend_embeddings import embed_batch, embed_single


class QdrantVectorDB:
    """Qdrant-based vector database with batch embedding support."""

    def __init__(
        self,
        collection_name: str,
        embedding_fn: Callable,
        vector_size: int = 768,
        distance: str = "cosine",
        persist_path: str = "./qdrant_db",
        mode: str = "local",
        url: str = None,
        api_key: str = None
    ):
        self.collection_name = collection_name
        self.embedding_fn = embedding_fn
        self.vector_size = vector_size

        # Initialize Qdrant client (local or cloud)
        if mode == "cloud" and url:
            sys.stderr.write(f"[Qdrant] Connecting to cloud: {url}" + "\n")
            self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
        else:
            sys.stderr.write(f"[Qdrant] Using local storage: {persist_path}" + "\n")
            self.client = QdrantClient(path=persist_path)

        # Map distance metric
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        self.distance = distance_map.get(distance, Distance.COSINE)

        # Create or get collection
        try:
            self._init_collection()
        except BaseException:
            self.client.close()
            raise

    def _init_collection(self):
        """Initialize Qdrant collection."""
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if not collection_exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                )
            )
            sys.stderr.write(f"[Qdrant] Created collection: {self.collection_name}" + "\n")
        else:
            sys.stderr.write(f"[Qdrant] Using existing collection: {self.collection_name}" + "\n")

    def add_documents_batch(self, documents: List[Dict[str, Any]], batch_size: int = 500):
        """
        Add documents in batches with batch embedding.

        Large batch size (500) for maximum embedding throughput.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        total = len(documents)
        sys.stderr.write(f"[Qdrant] Batch indexing {total} documents..." + "\n")

        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]

            # Extract content for batch embedding
            contents = [doc["content"] for doc in batch]

            # Batch embed (FAST)
            vectors = self._batch_embed(contents)
            if len(vectors) != len(batch):
                raise ValueError(f"Embedding count {len(vectors)} does not match document count {len(batch)}.")

            # Prepare points for Qdrant
            points = []
            for j, (doc, vector) in enumerate(zip(batch, vectors)):
                point_id = i + j
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=dict(doc)
                ))

            # Upsert batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            if (i + batch_size) % 500 == 0 or i + batch_size >= total:
                sys.stderr.write(f"[Qdrant] Indexed {min(i + batch_size, total)}/{total} documents..." + "\n")

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        return embed_batch(self, texts, "[Qdrant] Batch embedding via embed_documents failed")

    _embed_single = embed_single

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Vector similarity search."""
        query_vector = self._embed_single(query)

        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
        elif hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            )
            results = getattr(response, "points", response)
        else:
            raise AttributeError("QdrantClient has neither 'search' nor 'query_points' method")

        # Convert to standard format
        return [
            {"content": "", "score": hit.score, **hit.payload}
            for hit in results
        ]

    def count(self) -> int:
        """Get total document count."""
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count

    def clear(self):
        """Delete all documents from collection."""
        self.client.delete_collection(self.collection_name)
        self._init_collection()
        sys.stderr.write(f"[Qdrant] Cleared collection: {self.collection_name}" + "\n")
