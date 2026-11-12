"""Regressions from the 9 Nov retrieval audit: each failed on the code it audited."""

from types import SimpleNamespace
from unittest.mock import Mock

import pymupdf
import pytest
import rag.openrouter_pipeline as hosted
from rag import embeddings, fetch, server


@pytest.fixture(autouse=True)
def _answer_key(monkeypatch):
    """Synthesis is mocked here; without a key rag would return passages instead of calling it."""
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-only')

QWEN_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:question"


def write_pdf(path, pages):
    document = pymupdf.open()
    for _ in range(pages):
        document.new_page().insert_text((72, 72), "text")
    document.save(path)
    document.close()


def test_large_pdf_is_budgeted_by_pages_not_bytes(tmp_path, monkeypatch):
    pdf = tmp_path / "book.pdf"
    write_pdf(pdf, 2)
    with pdf.open("ab") as padding:  # Images and fonts inflate bytes, not text.
        padding.write(b"\0" * 400_000)
    monkeypatch.setenv("RAG_MCP_MAX_TOKENS", "50000")
    server._validate_rag_scope(pdf)


def test_pdf_with_too_many_pages_still_hits_the_budget(tmp_path, monkeypatch):
    pdf = tmp_path / "book.pdf"
    write_pdf(pdf, 3)
    monkeypatch.setenv("RAG_MCP_MAX_TOKENS", "2000")
    with pytest.raises(ValueError, match="token limit"):
        server._validate_rag_scope(pdf)


def run_multipart_query(tmp_path, monkeypatch, top):
    source = tmp_path / "notes.txt"
    source.write_text("notes")
    monkeypatch.setenv("RAG_MCP_RERANK_TOP_K", str(top))
    monkeypatch.setenv("RAG_MCP_CONTEXT_RADIUS", "0")
    calls = iter(range(100))

    def facet_passages(**_):
        group = next(calls)
        return [
            {"id": f"{group}-{i}", "content": f"{group}-{i}", "file_path": str(source)}
            for i in range(top)
        ]

    monkeypatch.setattr(fetch, "fetchExternalKnowledge", facet_passages)
    seen = {}

    def answer(query, passages, config):
        seen["passages"] = len(passages)
        return "answer"

    monkeypatch.setattr(hosted, "synthesize_answer", answer)
    server._query_knowledge_base(
        "What is alpha? How does beta work?", str(source), None
    )
    return seen["passages"]


@pytest.mark.parametrize("top", [3, 20])
def test_multipart_question_uses_the_configured_passage_count(
    tmp_path, monkeypatch, top
):
    assert run_multipart_query(tmp_path, monkeypatch, top) == top


def completion(reason, content="answer"):
    return {"choices": [{"finish_reason": reason, "message": {"content": content}}]}


def test_cut_off_answer_is_retried_with_more_room(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    caps = []

    def respond(path, payload, **_):
        caps.append(payload["max_tokens"])
        return completion("length" if len(caps) == 1 else "stop", "whole answer")

    monkeypatch.setattr(hosted, "_post", respond)
    assert (
        hosted.synthesize_answer(
            "question", [{"content": "evidence"}], {"answer_max_tokens": 700}
        )
        == "whole answer"
    )
    assert caps == [700, 2800]


def test_refused_answer_is_not_retried(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    post = Mock(return_value=completion("content_filter"))
    monkeypatch.setattr(hosted, "_post", post)
    with pytest.raises(RuntimeError, match="complete answer"):
        hosted.synthesize_answer("question", [{"content": "evidence"}], {})
    assert post.call_count == 1


def test_partial_evidence_is_answered_and_empty_evidence_still_abstains(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    post = Mock(return_value=completion("stop"))
    monkeypatch.setattr(hosted, "_post", post)
    hosted.synthesize_answer("question", [{"content": "evidence"}], {})
    instruction = post.call_args.args[1]["messages"][0]["content"]
    assert "answer only part of the question, answer that part" in instruction
    assert "If no passage addresses any part of the question" in instruction


def ollama(model):
    wrapper = embeddings.DirectOllamaGPUWrapper(model, "http://localhost:11434")
    wrapper._requests = Mock()
    wrapper._requests.post.return_value = Mock(status_code=200)
    wrapper._requests.post.return_value.json.return_value = {"embeddings": [[1.0, 0.0]]}
    wrapper._verify_gpu = Mock()
    wrapper.embed_query("question")
    return wrapper._requests.post.call_args.kwargs["json"]["input"]


@pytest.mark.parametrize(
    "model,expected",
    [
        ("qwen/qwen3-embedding-4b", QWEN_QUERY),
        ("nomic-embed-text", "question"),
    ],
)
def test_query_instruction_is_sent_only_to_qwen(model, expected):
    assert ollama(model) == [expected]


def sentence_transformer(prompts):
    wrapper = object.__new__(embeddings.STWrapper)
    encode = Mock(return_value=SimpleNamespace(tolist=lambda: None))
    encode.return_value = [Mock(tolist=lambda: [1.0])]
    wrapper.model = SimpleNamespace(encode=encode, prompts=prompts)
    wrapper.embed_query("question")
    return encode.call_args.kwargs


def test_sentence_transformer_uses_the_models_query_prompt():
    assert (
        sentence_transformer({"query": "Instruct: x", "document": ""})["prompt_name"]
        == "query"
    )


def test_sentence_transformer_without_query_prompt_sends_none():
    assert "prompt_name" not in sentence_transformer({})
