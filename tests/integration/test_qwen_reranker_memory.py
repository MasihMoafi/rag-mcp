"""Exercise real Qwen forward math without downloading pretrained weights."""

from types import SimpleNamespace

import pytest

from rag.reranking import rerank_documents


def test_qwen_scores_match_full_logits_without_unused_outputs():
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers.models.qwen3.modeling_qwen3')
    from transformers import Qwen3Config, Qwen3ForCausalLM

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=128,
        )).eval()

    class Tokenizer:
        pad_token_id = 0

        def convert_tokens_to_ids(self, token):
            return {'no': 1, 'yes': 2}[token]

        def encode(self, text, **kwargs):
            return [3, 4]

        def __call__(self, text, **kwargs):
            return {'input_ids': [5, 6, 7] if text.endswith('short') else [8] * 10}

    calls = []
    expected_scores = []

    def forward(input_ids, **kwargs):
        with torch.no_grad():
            full = model(input_ids, attention_mask=kwargs['attention_mask'])
            actual = model(input_ids, **kwargs)
        torch.testing.assert_close(actual.logits[:, -1], full.logits[:, -1])
        expected_scores.extend(full.logits[:, -1, [1, 2]].softmax(dim=-1)[:, 1].tolist())
        calls.append((kwargs, actual.logits.shape, actual.past_key_values))
        return actual

    docs = [{'id': 0, 'content': 'short'}, {'id': 1, 'content': 'longer'}]
    pipeline = SimpleNamespace(reranker_type='qwen3', reranker_tokenizer=Tokenizer(),
                               reranker_model=forward, device='cpu')
    result = rerank_documents(pipeline, docs, 'question', 1)
    assert len(calls) == 1, 'Qwen inference must run, not silently fall back'
    kwargs, shape, cache = calls[0]
    assert kwargs.get('logits_to_keep') == 1
    assert kwargs.get('use_cache') is False
    assert shape == (2, 1, 32)
    assert cache is None
    expected = sorted(zip(docs, expected_scores), key=lambda item: item[1], reverse=True)
    assert result == [expected[0][0]]
