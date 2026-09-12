from types import SimpleNamespace

import pytest

from rag.reranking import rerank_documents


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
