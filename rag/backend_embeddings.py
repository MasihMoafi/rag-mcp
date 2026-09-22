"""Embedding dispatch shared by the LanceDB and Qdrant adapters."""

import sys
from . import tracing


@tracing.traced('embedding.query')
def embed_single(self, text):
    tracing.annotate(model=getattr(self.embedding_fn, 'model_name', None), input_count=1)
    tracing.usage(None)
    if hasattr(self.embedding_fn, 'embed_query'):
        return self.embedding_fn.embed_query(text)
    if callable(self.embedding_fn):
        return self.embedding_fn(text)
    raise ValueError(f"Unsupported embedding function type: {type(self.embedding_fn)}")


@tracing.traced('embedding.documents')
def embed_batch(self, texts, failure_label):
    """Prefer batch methods; preserve ordered, bounded fallback for legacy providers."""
    tracing.annotate(model=getattr(self.embedding_fn, 'model_name', None), input_count=len(texts))
    tracing.usage(None)
    if hasattr(self.embedding_fn, 'embed_documents'):
        try:
            return self.embedding_fn.embed_documents(texts)
        except Exception as error:
            if not getattr(self.embedding_fn, 'retry_batch_as_queries', True):
                raise
            sys.stderr.write(f"{failure_label}: {type(error).__name__}\n")

    try:
        result = self.embedding_fn(texts)
        if isinstance(result, list) and len(result) == len(texts):
            return result
    except Exception:
        pass

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as executor:
        return list(executor.map(self._embed_single, texts))
