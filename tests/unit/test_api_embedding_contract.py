"""Provider responses must preserve the input-to-vector association."""

from types import SimpleNamespace

import pytest

from rag.embeddings import OpenAICompatibleWrapper


def wrapper_with_response(monkeypatch, items):
    wrapper = OpenAICompatibleWrapper.__new__(OpenAICompatibleWrapper)
    monkeypatch.setattr(wrapper, "_create_with_retry", lambda texts: SimpleNamespace(data=items))
    return wrapper


def item(index, vector):
    return SimpleNamespace(index=index, embedding=vector)


def test_vectors_follow_input_indices_not_response_order(monkeypatch):
    wrapper = wrapper_with_response(monkeypatch, [item(1, [0., 1.]), item(0, [1., 0.])])
    assert wrapper.embed_documents(["first", "second"]) == [[1., 0.], [0., 1.]]


@pytest.mark.parametrize("items", [
    [item(0, [1., 0.])],
    [item(0, [1., 0.]), item(0, [0., 1.])],
    [item(-1, [1., 0.]), item(1, [0., 1.])],
    [item(0, []), item(1, [])],
    [item(0, [1., 0.]), item(1, [1.])],
    [item(0, [float("nan"), 0.]), item(1, [0., 1.])],
    [item(0, [float("inf"), 0.]), item(1, [0., 1.])],
])
def test_invalid_response_is_rejected_before_storage(monkeypatch, items):
    wrapper = wrapper_with_response(monkeypatch, items)
    with pytest.raises(ValueError, match="embedding response"):
        wrapper.embed_documents(["first", "second"])


def test_empty_input_does_not_call_provider(monkeypatch):
    wrapper = wrapper_with_response(monkeypatch, [])
    monkeypatch.setattr(wrapper, "_create_with_retry", lambda _: pytest.fail("Unexpected API call"))
    assert wrapper.embed_documents([]) == []


def test_query_rejects_empty_vector(monkeypatch):
    wrapper = wrapper_with_response(monkeypatch, [item(0, [])])
    with pytest.raises(ValueError, match="embedding response"):
        wrapper.embed_query("question")


@pytest.mark.parametrize('status,expected_calls', [(401, 1), (400, 1), (429, 2), (503, 2)])
def test_sdk_retry_budget_is_not_multiplied(monkeypatch, status, expected_calls):
    import httpx
    import openai
    import time
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(status, json={'error': {'message': 'provider rejected request'}}, request=request)
    monkeypatch.setattr(time, 'sleep', lambda _: None)
    wrapper = OpenAICompatibleWrapper.__new__(OpenAICompatibleWrapper)
    wrapper.model_name = 'test-embedding'
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        wrapper.client = openai.OpenAI(api_key='test-only', max_retries=1, http_client=transport)
        with pytest.raises(openai.APIStatusError):
            wrapper.embed_documents(['evidence'])
    assert len(calls) == expected_calls


@pytest.mark.parametrize('backend_name', ['lancedb', 'qdrant'])
def test_api_batch_failure_does_not_fan_out_into_query_requests(monkeypatch, backend_name):
    from rag.lancedb_backend import LanceDBBackend
    from rag.qdrant_backend import QdrantVectorDB
    wrapper = OpenAICompatibleWrapper.__new__(OpenAICompatibleWrapper)
    calls = []
    def fail(texts):
        calls.append(texts)
        raise RuntimeError('provider unavailable')
    monkeypatch.setattr(wrapper, '_create_with_retry', fail)
    backend_type = LanceDBBackend if backend_name == 'lancedb' else QdrantVectorDB
    backend = backend_type.__new__(backend_type)
    backend.embedding_fn = wrapper
    with pytest.raises(RuntimeError, match='provider unavailable'):
        backend._batch_embed(['first', 'second'])
    assert calls == [['first', 'second']]
