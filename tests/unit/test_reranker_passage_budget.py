from types import SimpleNamespace

import pytest

from rag.reranking import rerank_documents


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
