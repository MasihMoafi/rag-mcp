import sys
import types

import pytest

from rag import embeddings


@pytest.mark.parametrize('flag', ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'])
@pytest.mark.parametrize('offline', [True, False])
def test_fastembed_respects_explicit_offline_setting(monkeypatch, flag, offline):
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        monkeypatch.delenv(name, raising=False)
    if offline:
        monkeypatch.setenv(flag, '1')
    seen = []
    monkeypatch.setitem(sys.modules, 'fastembed', types.SimpleNamespace(
        TextEmbedding=lambda **kwargs: seen.append((dict(kwargs), __import__('os').environ.get(flag)))))
    embeddings.FastEmbedWrapper('fixture')
    assert seen[0][0].get('local_files_only', False) is offline
    assert seen[0][1] == ('1' if offline else None)


def test_fastembed_reranker_respects_offline(monkeypatch):
    seen = []
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    module = types.ModuleType('fastembed.rerank.cross_encoder')
    module.TextCrossEncoder = lambda **kwargs: seen.append(kwargs)
    monkeypatch.setitem(sys.modules, 'fastembed.rerank.cross_encoder', module)
    embeddings.FastEmbedReranker('fixture')
    assert seen[0]['local_files_only'] is True
