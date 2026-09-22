"""Malformed or interrupted hosted output must not become a successful answer."""
import json
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


def test_synthesis_exception_does_not_leak_private_payload(monkeypatch, tmp_path):
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', 'off')
    monkeypatch.setattr(server, '_validate_rag_scope', lambda *a: None)
    monkeypatch.setattr(fetch, 'fetchExternalKnowledge', lambda **k: [{'content': 'evidence'}])
    monkeypatch.setattr(hosted, 'synthesize_answer', Mock(side_effect=RuntimeError('private-key and document')))
    response = json.loads(server.query_knowledge_base('question', str(tmp_path)))
    assert set(response) == {'error'}
    assert 'private-key' not in response['error']
    assert 'RuntimeError' in response['error']
