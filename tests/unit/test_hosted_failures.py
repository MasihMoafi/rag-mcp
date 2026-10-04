"""Malformed or interrupted hosted output must not become a successful answer."""
import email.message
import json
import urllib.error
from unittest.mock import Mock

import pytest

import server
from rag import fetch, openrouter_pipeline as hosted


@pytest.mark.parametrize('score', [True, float('nan'), float('inf'), -float('inf')])
def test_reranker_rejects_invalid_score(monkeypatch, score):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'fake')
    monkeypatch.setattr(hosted, '_post', lambda *a, **k: {
        'results': [{'index': 0, 'relevance_score': score}]})
    with pytest.raises(RuntimeError, match='invalid score'):
        hosted.OpenRouterReranker({}).predict([['question', 'evidence']])


@pytest.mark.parametrize('reason', [None, 'length', 'content_filter', 'error', 'tool_calls'])
def test_incomplete_synthesis_is_an_error(monkeypatch, reason):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'fake')
    monkeypatch.setattr(hosted, '_post', lambda *a, **k: {
        'choices': [{'finish_reason': reason, 'message': {'content': 'Incomplete answer'}}]})
    with pytest.raises(RuntimeError, match='complete answer'):
        hosted.synthesize_answer('question', [{'content': 'evidence'}], {})


def test_empty_retrieval_abstains_without_claiming_the_corpus_lacks_an_answer(monkeypatch):
    monkeypatch.setattr(hosted, '_post', lambda *a, **k: pytest.fail('No model call expected'))
    assert hosted.synthesize_answer('question', [], {}) == "I don't have enough information to answer that question."


@pytest.mark.parametrize('cite_sources', [False, True])
def test_hosted_prompt_scopes_abstention_to_supplied_evidence(monkeypatch, cite_sources):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'fake')
    captured = {}

    def response(path, payload, **kwargs):
        captured.update(payload)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': 'No supported answer.'}}]}

    monkeypatch.setattr(hosted, '_post', response)
    hosted.synthesize_answer('question', [{'content': 'unrelated evidence'}], {'cite_sources': cite_sources})
    instruction = captured['messages'][0]['content']
    assert 'indexed material' not in instruction
    assert 'do not have enough information to answer' in instruction
    assert 'Topical similarity alone is not evidence' in instruction
    assert 'Do not cite an abstention' in instruction
    assert ('Cite supporting passages' in instruction) is cite_sources


def test_synthesis_exception_does_not_leak_private_payload(monkeypatch, tmp_path):
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', 'off')
    monkeypatch.setattr(server, '_validate_rag_scope', lambda *a: None)
    monkeypatch.setattr(fetch, 'fetchExternalKnowledge', lambda **k: [{'content': 'evidence'}])
    monkeypatch.setattr(hosted, 'synthesize_answer', Mock(side_effect=RuntimeError('private-key and document')))
    response = json.loads(server.query_knowledge_base('question', str(tmp_path)))
    assert set(response) == {'error'}
    assert 'private-key' not in response['error']
    assert 'RuntimeError' in response['error']


def test_invalid_quote_error_is_distinct_from_provider_availability(monkeypatch, tmp_path):
    from rag.passage_answers import PassageAnswerError

    monkeypatch.setenv('RAG_MCP_TRACE_DIR', 'off')
    monkeypatch.setattr(server, '_validate_rag_scope', lambda *a: None)
    monkeypatch.setattr(fetch, 'fetchExternalKnowledge', lambda **k: [{'content': 'evidence'}])
    monkeypatch.setattr(hosted, 'synthesize_answer', Mock(side_effect=PassageAnswerError('private document')))
    response = json.loads(server.query_knowledge_base('question', str(tmp_path)))
    assert set(response) == {'error'}
    assert 'invalid or unsupported quote' in response['error']
    assert 'availability' not in response['error']
    assert 'private document' not in response['error']


def test_requested_book_references_receive_verified_pdf_pages(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'fake')
    captured = {}

    def response(path, payload, **kwargs):
        captured.update(payload)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': 'Book answer.'}}]}

    monkeypatch.setattr(hosted, '_post', response)
    hosted.synthesize_answer('Give a short quote and its exact PDF page number.', [
        {'content': 'First page.\fSecond page.', 'location': 'p.83-84',
         'file_path': '/private/book.pdf', 'score': 0.9},
        {'content': 'Other evidence.', 'location': ''},
    ], {})

    instruction = captured['messages'][0]['content']
    evidence = captured['messages'][1]['content']
    assert 'Do not mention retrieval, passages, files, sources' not in instruction
    assert 'PDF page 83' in evidence and 'PDF page 84' in evidence
    assert 'Printed page number: unavailable' in evidence
    assert 'Source location: unavailable' in evidence
    assert '/private/book.pdf' not in evidence
    assert '0.9' not in evidence


def test_invalid_pdf_location_cannot_supply_a_page_number(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'fake')
    captured = {}

    def response(path, payload, **kwargs):
        captured.update(payload)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': 'Unavailable.'}}]}

    monkeypatch.setattr(hosted, '_post', response)
    hosted.synthesize_answer('Give the PDF page number.', [
        {'content': 'Evidence.', 'location': 'p.0', 'page_number': 900},
    ], {})

    assert 'Source location: unavailable' in captured['messages'][1]['content']
    assert '900' not in captured['messages'][1]['content']


def test_pdf_page_breaks_require_current_source_text(monkeypatch):
    from rag.parsers import pdf

    monkeypatch.setattr(pdf, 'extract_pdf_text', lambda path: 'Cover\fFirst.\fSecond.')
    passage = {'file_path': '/book.pdf', 'location': 'p.2-3',
               'start_offset': 6, 'end_offset': 20}
    content = 'First. Second.'
    evidence = hosted._answer_evidence({**passage, 'content': content}, source_details=True, pdf_cache={})
    assert '[PDF page 2]\nFirst.' in evidence
    assert '[PDF page 3]\nSecond.' in evidence

    monkeypatch.setattr(pdf, 'extract_pdf_text', lambda path: 'Cover\fChanged.\fSecond.')
    evidence = hosted._answer_evidence({**passage, 'content': content}, source_details=True, pdf_cache={})
    assert 'Exact page within range: unavailable' in evidence
    assert '[PDF page 2]' not in evidence


class _Body:
    """The context-manager response urlopen returns."""
    def __init__(self, value):
        self._raw = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *args):
        return self._raw


def _http_error(code, retry_after=None):
    headers = email.message.Message()
    if retry_after is not None:
        headers['Retry-After'] = str(retry_after)
    return urllib.error.HTTPError('https://openrouter.ai/api/v1/rerank', code, 'error', headers, None)


def _stub_network(monkeypatch, outcomes):
    """Make urlopen yield each outcome in turn: an exception to raise, or a value to return."""
    calls, sleeps = [], []
    def urlopen(request, timeout):
        outcome = outcomes[min(len(calls), len(outcomes) - 1)]
        calls.append(request.full_url)
        if isinstance(outcome, BaseException):
            raise outcome
        return _Body(outcome)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'fake')
    monkeypatch.setattr(hosted.urllib.request, 'urlopen', urlopen)
    monkeypatch.setattr(hosted.time, 'sleep', sleeps.append)
    return calls, sleeps


SCORED = {'results': [{'index': 0, 'relevance_score': 0.9}, {'index': 1, 'relevance_score': 0.1}]}


def test_reranker_retries_a_dropped_connection_then_scores(monkeypatch):
    calls, sleeps = _stub_network(monkeypatch, [urllib.error.URLError('connection reset'), TimeoutError(), SCORED])
    assert hosted.OpenRouterReranker({}).predict([['q', 'a'], ['q', 'b']]) == [0.9, 0.1]
    assert len(calls) == 3 and sleeps == [0.5, 1.5]


def test_reranker_honours_a_short_retry_after_and_caps_a_long_one(monkeypatch):
    calls, sleeps = _stub_network(monkeypatch, [_http_error(429, retry_after=2), _http_error(503, retry_after=600), SCORED])
    assert hosted.OpenRouterReranker({}).predict([['q', 'a'], ['q', 'b']]) == [0.9, 0.1]
    assert sleeps == [2.0, 5.0]


def test_reranker_does_not_retry_a_client_error(monkeypatch):
    calls, sleeps = _stub_network(monkeypatch, [_http_error(401)])
    with pytest.raises(RuntimeError, match='HTTP 401'):
        hosted.OpenRouterReranker({}).predict([['q', 'a'], ['q', 'b']])
    assert len(calls) == 1 and sleeps == []


def test_reranker_gives_up_after_its_bounded_retries(monkeypatch):
    calls, sleeps = _stub_network(monkeypatch, [_http_error(503)])
    with pytest.raises(RuntimeError, match='HTTP 503'):
        hosted.OpenRouterReranker({}).predict([['q', 'a'], ['q', 'b']])
    assert len(calls) == 3 and sleeps == [0.5, 1.5]


def test_a_synthesized_answer_is_never_requested_twice(monkeypatch):
    calls, sleeps = _stub_network(monkeypatch, [_http_error(503)])
    with pytest.raises(RuntimeError, match='HTTP 503'):
        hosted._post('/chat/completions', {'model': 'm', 'messages': [{'content': 'question'}]}, key='fake', timeout=1)
    assert len(calls) == 1 and sleeps == []
