"""Provider responses must preserve the input-to-vector association."""

from types import SimpleNamespace

import pytest

from rag.embeddings import OpenAICompatibleWrapper


def wrapper_with_response(monkeypatch, items):
    wrapper = OpenAICompatibleWrapper.__new__(OpenAICompatibleWrapper)
    monkeypatch.setattr(wrapper, "_create_with_retry", lambda texts: SimpleNamespace(data=items))
    return wrapper


def item(index, vector):
    return SimpleNamespace(index=index, embedding=vector)


def test_vectors_follow_input_indices_not_response_order(monkeypatch):
    wrapper = wrapper_with_response(monkeypatch, [item(1, [0., 1.]), item(0, [1., 0.])])
    assert wrapper.embed_documents(["first", "second"]) == [[1., 0.], [0., 1.]]


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


def test_query_rejects_empty_vector(monkeypatch):
    wrapper = wrapper_with_response(monkeypatch, [item(0, [])])
    with pytest.raises(ValueError, match="embedding response"):
        wrapper.embed_query("question")
