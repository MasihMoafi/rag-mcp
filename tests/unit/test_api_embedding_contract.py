"""Provider responses must preserve the input-to-vector association."""

from types import SimpleNamespace

import pytest

from rag.embeddings import OpenAICompatibleWrapper


@pytest.mark.parametrize('operation,count', [('batch', 511), ('batch', 513), ('query', 0), ('query', 2)])
def test_ollama_rejects_wrong_count_before_more_requests(operation, count):
    from unittest.mock import Mock
    from rag.embeddings import DirectOllamaGPUWrapper

    wrapper = DirectOllamaGPUWrapper('fixture', 'http://localhost:11434')
    wrapper._requests = Mock()
    wrapper._requests.post.return_value = Mock(status_code=200)
    wrapper._requests.post.return_value.json.return_value = {'embeddings': [[1., 0.]] * count}
    wrapper._verify_gpu = Mock()
    with pytest.raises(ValueError, match='embedding count'):
        if operation == 'batch':
            wrapper.embed_documents(['document'] * 513)
        else:
            wrapper.embed_query('question')
    assert wrapper._requests.post.call_count == 1
    wrapper._verify_gpu.assert_not_called()


def test_ollama_valid_batches_preserve_vectors_and_timeouts():
    from unittest.mock import Mock
    from rag.embeddings import DirectOllamaGPUWrapper

    wrapper = DirectOllamaGPUWrapper('fixture', 'http://localhost:11434')
    wrapper._requests = Mock()
    def respond(url, *, json, timeout):
        vectors = [[float(text), 1.] for text in json['input']]
        return SimpleNamespace(status_code=200, json=lambda: {'embeddings': vectors})
    wrapper._requests.post.side_effect = respond
    wrapper._verify_gpu = Mock()
    assert wrapper.embed_documents([str(i) for i in range(513)]) == [[float(i), 1.] for i in range(513)]
    assert wrapper.embed_query('514') == [514., 1.]
    calls = wrapper._requests.post.call_args_list
    assert [len(call.kwargs['json']['input']) for call in calls] == [512, 1, 1]
    assert [call.kwargs['timeout'] for call in calls] == [180, 180, 60]
    assert wrapper._verify_gpu.call_count == 2


def wrapper_with_response(monkeypatch, items):
    wrapper = OpenAICompatibleWrapper.__new__(OpenAICompatibleWrapper)
    monkeypatch.setattr(wrapper, "_create_with_retry", lambda texts: SimpleNamespace(data=items))
    return wrapper


@pytest.mark.parametrize('vectors', [[[]], [[float('nan')]], [[float('inf')]], [[1.], [1., 2.]]])
def test_ollama_rejects_invalid_vector_values(vectors):
    from unittest.mock import Mock
    from rag.embeddings import DirectOllamaGPUWrapper

    wrapper = DirectOllamaGPUWrapper('fixture', 'http://localhost:11434')
    wrapper._requests = Mock()
    wrapper._requests.post.return_value = Mock(status_code=200)
    wrapper._requests.post.return_value.json.return_value = {'embeddings': vectors}
    wrapper._verify_gpu = Mock()
    with pytest.raises(ValueError, match='empty, inconsistent or non-finite'):
        wrapper.embed_documents(['document'] * len(vectors))
    if len(vectors) == 1:
        with pytest.raises(ValueError, match='empty, inconsistent or non-finite'):
            wrapper.embed_query('question')
    wrapper._verify_gpu.assert_not_called()


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


def test_unknown_provider_does_not_load_a_local_model(monkeypatch):
    from rag import embeddings
    monkeypatch.setattr(embeddings, 'STWrapper', lambda *args: pytest.fail('Unexpected local model load'))
    with pytest.raises(ValueError, match='Unknown embedding provider'):
        embeddings.build_embedder({'embed_provider': 'openai_typo'})


def test_unknown_backend_is_rejected_before_model_initialization(monkeypatch):
    from rag import core
    monkeypatch.setattr(core, 'build_embedder', lambda *args, **kwargs: pytest.fail('Unexpected model load'))
    with pytest.raises(ValueError, match='Unknown retrieval backend'):
        core.RAGPipeline({'backend': 'lance_typo', 'persist_dir': '/unused'})


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
