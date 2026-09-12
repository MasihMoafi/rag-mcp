"""Reject invalid retrieval limits before allocating models or index storage."""
from unittest.mock import Mock

import pytest

from rag import core


@pytest.mark.parametrize("key", ["top_k", "rerank_top_k"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "5", None])
def test_invalid_retrieval_limits_fail_before_index_initialization(monkeypatch, key, value):
    initialize = Mock()
    monkeypatch.setattr(core, "open_or_build", initialize)
    with pytest.raises(ValueError, match=key):
        core.RAGPipeline({key: value})
    initialize.assert_not_called()


def test_valid_limits_and_defaults_leave_caller_config_unchanged(monkeypatch):
    initialize = Mock()
    monkeypatch.setattr(core, "open_or_build", initialize)
    config = {"top_k": 20, "rerank_top_k": 5}
    core.RAGPipeline(config)
    assert config == {"top_k": 20, "rerank_top_k": 5}
    core.RAGPipeline({})
    assert initialize.call_count == 2
