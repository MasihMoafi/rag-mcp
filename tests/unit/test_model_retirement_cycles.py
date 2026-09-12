"""Retiring a model must release unreachable cycles, not wait for later traffic."""
import gc
import weakref

import pytest

from rag.core import RAGPipeline
from rag.query_lifecycle import QueryLifecycle


@pytest.mark.parametrize('attribute', ['reranker', 'reranker_model'])
def test_close_collects_retired_model_cycle(attribute):
    class Model:
        pass

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._queries = QueryLifecycle()
    pipeline.backend_type = 'lancedb'
    pipeline.vector_db = pipeline.embedding_fn = None
    model = Model()
    model.cycle = model
    reference = weakref.ref(model)
    setattr(pipeline, attribute, model)
    del model
    enabled = gc.isenabled()
    gc.disable()
    try:
        pipeline.close()
        assert reference() is None
        pipeline.close()
    finally:
        if enabled:
            gc.enable()
        gc.collect()
