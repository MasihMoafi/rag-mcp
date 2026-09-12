from types import SimpleNamespace

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
