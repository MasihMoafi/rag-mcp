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


def test_ollama_reports_aggregate_embedding_tokens_without_response_content(monkeypatch, tmp_path):
    from unittest.mock import Mock
    from rag import tracing
    from rag.embeddings import DirectOllamaGPUWrapper
    from rag.trace_reader import read_trace

    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(tmp_path))
    wrapper = DirectOllamaGPUWrapper('fixture', 'http://localhost:11434')
    wrapper._requests = Mock()
    wrapper._verify_gpu = Mock()
    counts = iter((1024, 3, 2))

    def respond(url, *, json, timeout):
        return SimpleNamespace(status_code=200, json=lambda: {
            'embeddings': [[1., 0.]] * len(json['input']),
            'prompt_eval_count': next(counts),
            'private_marker': 'must-not-appear',
        })

    wrapper._requests.post.side_effect = respond
    trace_id = 'b' * 32
    with tracing.request_trace(trace_id):
        with tracing.span('embedding.documents'):
            assert len(wrapper.embed_documents(['document'] * 513)) == 513
        with tracing.span('embedding.query'):
            assert wrapper.embed_query('question') == [1., 0.]
    trace = read_trace(trace_id)
    usages = [span['attributes']['usage'] for span in trace['spans'][1:]]
    assert [usage['input_tokens'] for usage in usages] == [1027, 2]
    assert all(usage['output_tokens'] is None and usage['cost_usd'] is None for usage in usages)
    assert 'must-not-appear' not in str(trace)


@pytest.mark.parametrize('count', [None, True, -1, 2**60])
def test_ollama_incomplete_or_invalid_usage_stays_unknown(monkeypatch, tmp_path, count):
    from unittest.mock import Mock
    from rag import tracing
    from rag.embeddings import DirectOllamaGPUWrapper
    from rag.trace_reader import read_trace

    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(tmp_path))
    wrapper = DirectOllamaGPUWrapper('fixture', 'http://localhost:11434')
    wrapper._requests = Mock()
    wrapper._verify_gpu = Mock()
    counts = iter((10, count))

    def respond(url, *, json, timeout):
        reported = next(counts)
        payload = {'embeddings': [[1., 0.]] * len(json['input'])}
        if reported is not None:
            payload['prompt_eval_count'] = reported
        return SimpleNamespace(status_code=200, json=lambda: payload)

    wrapper._requests.post.side_effect = respond
    trace_id = 'c' * 32
    with tracing.request_trace(trace_id):
        with tracing.span('embedding.documents'):
            assert len(wrapper.embed_documents(['document'] * 513)) == 513
    usage = read_trace(trace_id)['spans'][1]['attributes'].get('usage')
    assert usage is None or usage['input_tokens'] is None


def test_ollama_rejects_dimension_change_between_batches():
    from unittest.mock import Mock
    from rag.embeddings import DirectOllamaGPUWrapper

    wrapper = DirectOllamaGPUWrapper('fixture', 'http://localhost:11434')
    wrapper._requests = Mock()
    wrapper._requests.post.side_effect = [
        SimpleNamespace(status_code=200, json=lambda: {'embeddings': [[1., 0.]] * 512}),
        SimpleNamespace(status_code=200, json=lambda: {'embeddings': [[1., 0., 0.]]}),
    ]
    wrapper._verify_gpu = Mock()
    with pytest.raises(ValueError, match='dimensions changed between batches'):
        wrapper.embed_documents(['document'] * 513)
    assert wrapper._requests.post.call_count == 2
    wrapper._verify_gpu.assert_not_called()


def wrapper_with_response(monkeypatch, items):
    wrapper = OpenAICompatibleWrapper.__new__(OpenAICompatibleWrapper)
    monkeypatch.setattr(wrapper, "_create_with_retry", lambda texts, *_: SimpleNamespace(data=items))
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


def test_embedding_batch_records_only_provider_reported_usage(monkeypatch, tmp_path):
    from rag import tracing
    from rag.trace_reader import read_trace
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(tmp_path))
    wrapper = OpenAICompatibleWrapper.__new__(OpenAICompatibleWrapper)
    monkeypatch.setattr(wrapper, '_create_with_retry', lambda texts: SimpleNamespace(
        data=[item(0, [1., 0.])],
        usage=SimpleNamespace(prompt_tokens=5, total_tokens=5, cost=0.0002,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=2),
                              private_marker='must-not-appear')))
    trace_id = 'a' * 32
    with tracing.request_trace(trace_id):
        with tracing.span('embedding.documents'):
            assert wrapper.embed_documents(['first']) == [[1., 0.]]
    trace = read_trace(trace_id)
    usage = trace['spans'][1]['attributes']['usage']
    assert usage['input_tokens'] == 5
    assert usage['output_tokens'] is None
    assert usage['cost_usd'] == 0.0002
    assert usage['cached_tokens'] == 2
    assert trace['accounting']['reported_cost_subtotal_usd'] == 0.0002
    assert 'must-not-appear' not in str(trace)


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


def test_duplicate_inputs_are_embedded_once_without_normalizing(monkeypatch):
    wrapper = OpenAICompatibleWrapper.__new__(OpenAICompatibleWrapper)
    calls = []
    def respond(texts):
        calls.append(list(texts))
        return SimpleNamespace(data=[item(i, [float(i), 1.]) for i in reversed(range(len(texts)))])
    monkeypatch.setattr(wrapper, '_create_with_retry', respond)
    vectors = wrapper.embed_documents(['کتاب', 'other', 'کتاب', ' کتاب'])
    assert calls == [['کتاب', 'other', ' کتاب']]
    assert vectors == [[0., 1.], [1., 1.], [0., 1.], [2., 1.]]
    vectors[0][0] = 99
    assert vectors[2] == [0., 1.]


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


# A 429 is rejected, not billed: it is waited out within a bounded budget.
@pytest.mark.parametrize('status,expected_calls', [
    (401, 1), (400, 1), (429, 2 * (len(OpenAICompatibleWrapper.RATE_LIMIT_WAITS) + 1)), (503, 2)])
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


def _rate_limited(retry_after=None):
    import httpx
    from openai import RateLimitError

    headers = {'retry-after': retry_after} if retry_after else {}
    response = httpx.Response(429, headers=headers, request=httpx.Request('POST', 'https://openrouter.test/v1/embeddings'))
    return RateLimitError('rate limited', response=response, body=None)


def test_api_embedding_waits_out_provider_rate_limits(monkeypatch):
    from rag import embeddings

    waits = []
    monkeypatch.setattr(embeddings.time, 'sleep', waits.append)
    wrapper = OpenAICompatibleWrapper('fixture', api_key='key', base_url='https://openrouter.test/v1')
    ok = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1., 0.])], usage=None)
    calls = iter([_rate_limited('3'), _rate_limited(), ok])

    def create(**kwargs):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    wrapper.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    assert wrapper.embed_documents(['document']) == [[1., 0.]]
    assert waits == [3., 4]


def test_api_embedding_gives_up_after_bounded_rate_limit_waits(monkeypatch):
    from openai import RateLimitError
    from rag import embeddings

    waits = []
    monkeypatch.setattr(embeddings.time, 'sleep', waits.append)
    wrapper = OpenAICompatibleWrapper('fixture', api_key='key', base_url='https://openrouter.test/v1')

    def create(**kwargs):
        raise _rate_limited('999')

    wrapper.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    with pytest.raises(RateLimitError):
        wrapper.embed_documents(['document'])
    assert len(waits) == len(OpenAICompatibleWrapper.RATE_LIMIT_WAITS)
    assert max(waits) <= 60


def test_query_embedding_fails_fast_and_resends_only_a_stall(monkeypatch):
    from threading import Event
    from openai import RateLimitError
    from rag import embeddings

    waits = []
    monkeypatch.setattr(embeddings.time, 'sleep', waits.append)
    wrapper = OpenAICompatibleWrapper('fixture', api_key='key', base_url='https://openrouter.test/v1')
    calls = []

    def limited(**kwargs):
        calls.append(kwargs['input'])
        raise _rate_limited('30')

    wrapper.client = SimpleNamespace(embeddings=SimpleNamespace(create=limited))
    with pytest.raises(RateLimitError):
        wrapper.embed_documents(['query'], resend_after=(0.05, 0.1))
    assert waits == [] and len(calls) == 1

    stalled, ok = Event(), SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1., 0.])], usage=None)

    def first_stalls(**kwargs):
        calls.append(kwargs['input'])
        if len(calls) == 2:
            stalled.wait(5)
        return ok

    wrapper.client = SimpleNamespace(embeddings=SimpleNamespace(create=first_stalls))
    assert wrapper.embed_documents(['query'], resend_after=(0.05, 0.1)) == [[1., 0.]]
    stalled.set()
    assert len(calls) == 3
