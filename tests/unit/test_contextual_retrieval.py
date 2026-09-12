from types import SimpleNamespace

from rag import fetch
from rag.core import RAGPipelineV2
from rag.contextual_retrieval import AnthropicContextualizer, select_document_context


def test_enabled_contextual_retrieval_prepends_generated_context():
    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    pipeline.config = {"use_contextual": True}
    calls = []

    def contextualizer(chunk, document_context):
        calls.append((chunk, document_context))
        return "This chunk belongs to the incident report."

    pipeline.contextualizer = contextualizer

    result = pipeline._add_context_to_chunk(
        "The engineering team fixed it.",
        "Full incident report.",
    )

    assert result == "This chunk belongs to the incident report.\nThe engineering team fixed it."
    assert calls == [("The engineering team fixed it.", "Full incident report.")]


def test_disabled_contextual_retrieval_keeps_original_chunk_without_calling_contextualizer():
    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    pipeline.config = {"use_contextual": False}

    def unexpected_call(*_args):
        raise AssertionError("contextualizer should not run when disabled")

    pipeline.contextualizer = unexpected_call

    assert pipeline._add_context_to_chunk("original", "source") == "original"


def test_contextualizer_failure_falls_back_to_original_chunk(capsys):
    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    pipeline.config = {"use_contextual": True}

    def failing_contextualizer(_chunk, _document_context):
        raise RuntimeError("API unavailable")

    pipeline.contextualizer = failing_contextualizer

    assert pipeline._add_context_to_chunk("original", "source") == "original"
    assert "API unavailable" in capsys.readouterr().err


def test_large_sources_use_start_and_previous_chunks_for_context():
    chunks = ["abstract", "first", "second", "target", "later"]

    context = select_document_context(
        "long source",
        chunks,
        3,
        max_source_chars=4,
        start_chunks=1,
        previous_chunks=2,
    )

    assert context == "abstract\n\nfirst\n\nsecond"


def test_anthropic_contextualizer_sends_prompt_and_returns_text():
    calls = []

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="  Situated context.  ")])

    contextualizer = AnthropicContextualizer(
        model="claude-haiku-4-5",
        max_tokens=128,
        timeout=30,
        client=SimpleNamespace(messages=Messages()),
    )

    assert contextualizer("chunk text", "document context") == "Situated context."
    assert calls[0]["model"] == "claude-haiku-4-5"
    assert calls[0]["max_tokens"] == 128
    assert calls[0]["temperature"] == 0
    assert "chunk text" in calls[0]["messages"][0]["content"]
    assert "document context" in calls[0]["messages"][0]["content"]


def test_contextualized_indexing_keeps_original_content_for_evidence(tmp_path):
    document = tmp_path / "notes.txt"
    document.write_text("The engineering team fixed the incident.\n\nThe report is complete.")

    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    indexed = []
    pipeline.config = {
        "document_paths": [str(document)],
        "chunk_size": 700,
        "chunk_overlap": 0,
        "use_contextual": True,
        "contextual_source_max_chars": 12000,
        "contextual_start_chunks": 2,
        "contextual_prev_chunks": 2,
    }
    pipeline.contextualizer = lambda chunk, context: f"Located in the report: {context[:20]}"
    pipeline.vector_db = SimpleNamespace(add_documents_batch=lambda docs, batch_size: indexed.extend(docs))
    pipeline.bm25_index = None
    pipeline._save_database = lambda _docs: None

    pipeline._index_documents()

    assert indexed
    assert indexed[0]["content"].startswith("Located in the report:")
    assert indexed[0]["original_content"] == "The engineering team fixed the incident.\n\nThe report is complete."


def test_contextual_setting_change_forces_a_new_index(tmp_path, monkeypatch):
    document = tmp_path / "facts.txt"
    document.write_text("The launch codename is Cedar Lantern.")
    monkeypatch.setenv("RAG_MCP_DATA_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("RAG_MCP_EMBED_PROVIDER", "none")
    monkeypatch.setenv("RAG_MCP_CONTEXTUAL_RETRIEVAL", "off")
    fetch.reset_pipelines()
    captured = []

    class FakePipeline:
        def __init__(self, *, config):
            self.config = config
            captured.append(config)

    monkeypatch.setattr(fetch, "RAGPipelineV2", FakePipeline)
    fetch.get_rag_pipeline_v2(str(document))
    fetch.get_rag_pipeline_v2(str(document))

    monkeypatch.setenv("RAG_MCP_CONTEXTUAL_RETRIEVAL", "on")
    fetch.get_rag_pipeline_v2(str(document))

    assert len(captured) == 2
    assert captured[0]["use_contextual"] is False
    assert captured[1]["use_contextual"] is True
    assert captured[1]["force_reindex"] is True
