from types import SimpleNamespace
import json

import numpy as np
import pytest

from rag import tracing
from rag.reranking import rerank_documents


def test_reranker_uses_basename_without_mutating_source_records():
    from copy import deepcopy
    docs = [{'content': 'Original evidence.', 'file_path': '/private/client/Napoleon.pdf', 'location': 'p.3'},
            {'content': 'Other evidence.'}]
    original = deepcopy(docs)
    received = []

    def predict(pairs):
        received.extend(pairs)
        return [2.0, 1.0]

    pipeline = SimpleNamespace(reranker_type='cross-encoder', reranker=SimpleNamespace(predict=predict))
    results = rerank_documents(pipeline, docs, 'question', 2)
    assert received == [['question', 'Source: Napoleon.pdf\nOriginal evidence.'],
                        ['question', 'Other evidence.']]
    assert docs == original
    assert results[0] is docs[0]
    assert results[1] is docs[1]


def test_reranker_trace_selected_scores_follow_model_without_mutating_documents(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(tmp_path / 'traces'))
    docs = [{'id': 'first', 'content': 'private-first', 'score': 12.0},
            {'id': 'second', 'content': 'private-second', 'score': 3.0}]
    pipeline = SimpleNamespace(reranker_type='cross-encoder',
                               reranker=SimpleNamespace(predict=lambda pairs: np.array([0.2, 0.9], dtype=np.float32)))
    with tracing.request_trace():
        with tracing.span('selection.rerank'):
            selected = rerank_documents(pipeline, docs, 'private-question', 2)
    assert selected == docs[::-1]
    assert selected[0] is docs[1] and selected[1] is docs[0]
    assert [doc['score'] for doc in docs] == [12.0, 3.0]
    raw = next((tmp_path / 'traces').glob('trace-*.json')).read_text()
    span = next(item for item in json.loads(raw)['spans'] if item['operation'] == 'selection.rerank')
    assert span['attributes']['score_kind'] == 'reranker'
    assert [row['score'] for row in span['attributes']['selected']] == pytest.approx([0.9, 0.2])
    assert 'private-' not in raw


def test_local_reranker_failure_marks_retained_scores_as_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(tmp_path / 'traces'))
    docs = [{'id': 'first', 'content': 'private-first', 'score': 12.0},
            {'id': 'second', 'content': 'private-second', 'score': 3.0}]
    def fail(_pairs):
        raise RuntimeError('private-failure')
    pipeline = SimpleNamespace(reranker_type='cross-encoder',
                               reranker=SimpleNamespace(predict=fail))
    with tracing.request_trace():
        with tracing.span('selection.rerank'):
            selected = rerank_documents(pipeline, docs, 'private-question', 2)
    assert selected == docs
    span = next(item for item in json.loads(next((tmp_path / 'traces').glob('trace-*.json')).read_text())['spans']
                if item['operation'] == 'selection.rerank')
    assert span['status'] == 'skipped'
    assert span['attributes']['skipped_reason'] == 'reranker_failed'
    assert span['attributes']['score_kind'] == 'retained_source'
    assert [row['score'] for row in span['attributes']['selected']] == [12.0, 3.0]


@pytest.mark.parametrize('k', [2, 5])
@pytest.mark.parametrize('failure', [False, True])
def test_reranker_orders_candidates_even_when_all_fit(k, failure):
    docs = [{'id': i, 'content': f'Passage {i}'} for i in range(2)]
    calls = []

    def predict(pairs):
        calls.append(pairs)
        if failure:
            raise RuntimeError('unavailable')
        return [0.1, 0.9]

    pipeline = SimpleNamespace(reranker_type='cross-encoder',
                               reranker=SimpleNamespace(predict=predict))
    result = rerank_documents(pipeline, docs, 'question', k)
    assert calls == [[['question', doc['content']] for doc in docs]]
    expected = docs if failure else docs[::-1]
    assert result == expected
    assert all(actual is original for actual, original in zip(result, expected))


def test_cross_encoder_receives_answer_beyond_512_characters():
    answer = 'The release code is ORCHID.'
    relevant = {'id': 'answer', 'content': 'Background details. ' * 40 + answer}
    distractor = {'id': 'other', 'content': 'No release code here.'}
    received = []

    def predict(pairs):
        received.extend(pairs)
        return [float(answer in passage) for _, passage in pairs]

    pipeline = SimpleNamespace(reranker_type='cross-encoder',
                               reranker=SimpleNamespace(predict=predict))
    assert rerank_documents(pipeline, [distractor, relevant], 'Release code?', 1) == [relevant]
    assert received[1][1] == relevant['content']


@pytest.mark.parametrize('setting', ['llm', 'LLM'])
def test_unsupported_llm_setting_fails_before_resource_initialization(monkeypatch, setting):
    from rag import core

    monkeypatch.setattr(core, 'open_or_build', lambda *a, **k: pytest.fail('unsupported setting reached storage'))
    with pytest.raises(ValueError, match='LLM reranking is not implemented'):
        core.RAGPipeline({'reranker_type': setting})


@pytest.mark.parametrize('setting', ['cross_encoderr', 'some/model-name', '', 42])
def test_unknown_reranker_setting_fails_before_resource_initialization(monkeypatch, setting):
    from rag import core

    monkeypatch.setattr(core, 'open_or_build', lambda *a, **k: pytest.fail('invalid setting reached storage'))
    with pytest.raises(ValueError, match='Unknown reranker type'):
        core.RAGPipeline({'reranker_type': setting})


@pytest.mark.parametrize('setting', ['AUTO', 'CROSS-ENCODER', 'FASTEMBED', 'disabled', 'NONE', 'OFF'])
def test_supported_reranker_settings_reach_initialization(monkeypatch, setting):
    from rag import core

    received = []
    monkeypatch.setattr(core, 'open_or_build', lambda config, initialize: received.append(config))
    core.RAGPipeline({'reranker_type': setting})
    assert received[0]['reranker_type'] == setting


@pytest.mark.parametrize('scores', [[], [0.9], [0.1, 0.2, 0.3, 0.4],
                                   [0.1, float('nan'), 0.9],
                                   [0.1, float('inf'), 0.9],
                                   [0.1, float('-inf'), 0.9]])
def test_invalid_reranker_scores_preserve_fallback_passages(scores, capsys):
    docs = [{'id': i, 'content': f'Passage {i}'} for i in range(3)]
    pipeline = SimpleNamespace(reranker_type='cross-encoder',
        reranker=SimpleNamespace(predict=lambda pairs: scores))
    assert rerank_documents(pipeline, docs, 'question', 2) == docs[:2]
    assert 'ValueError' in capsys.readouterr().out
