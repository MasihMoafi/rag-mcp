"""Source-boundary and citation regressions from the production-RAG book query."""

from types import SimpleNamespace
import json

import pytest

from rag.core import RAGPipelineV2, chunk_document
from rag.lancedb_backend import LanceDBBackend
from rag import fetch
from rag.ingestion import build_documents


@pytest.mark.parametrize("separator", ["\n\n", "\n\n\n", "\n  \n"])
def test_chunks_are_contiguous_source_slices_with_whole_words(separator):
    paragraphs = [f"Paragraph {i} explains document structure and preserves important context." for i in range(12)]
    source = separator.join(paragraphs)
    chunks = chunk_document(source, "chapter.md", chunk_size=150, chunk_overlap=25)
    covered = set()
    for chunk in chunks:
        content = chunk["content"]
        start = source.find(content)
        assert start >= 0, repr(content)
        end = start + len(content)
        assert start == 0 or source[start - 1].isspace()
        assert end == len(source) or source[end].isspace()
        covered.update(range(start, end))
    assert all(i in covered for i, char in enumerate(source) if not char.isspace())


def test_markdown_preamble_does_not_disable_headings():
    source = "Preamble.\n\n# Book\n\nIntroduction.\n\n## Chunking\n\nKeep the evidence intact."
    chunks = chunk_document(source, "chapter.md")
    assert chunks[-1]["heading"] == "Book > Chunking"
    assert all(chunk["content"] in source for chunk in chunks)


def test_long_prose_splits_at_words_and_keeps_all_text():
    source = " ".join(f"evidence{i}" for i in range(100))
    chunks = chunk_document(source, "chapter.txt", chunk_size=100, chunk_overlap=20)
    words = {word for chunk in chunks for word in chunk["content"].split()}
    assert words == set(source.split())
    assert all(chunk["content"] in source for chunk in chunks)


def test_overlap_does_not_prepend_a_fragment_of_a_short_paragraph():
    source = "\n\n".join(f"Paragraph {i} explains document structure and preserves important context." for i in range(12))
    chunks = chunk_document(source, "chapter.md", chunk_size=150, chunk_overlap=25)
    assert all(chunk["content"].startswith("Paragraph ") for chunk in chunks)


def test_long_word_after_short_prefix_is_not_split_unnecessarily():
    source = "short " + "a" * 98 + " trailing words"
    chunks = chunk_document(source, "a.txt", chunk_size=100, chunk_overlap=0)
    assert [word for chunk in chunks for word in chunk["content"].split()] == source.split()


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunk_config_is_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_document("A source document", "a.txt", size, overlap)


def test_ingestion_metadata_survives_lancedb_serialization(tmp_path):
    source = tmp_path / "chapter.md"
    source.write_text("# Book\n\nFirst paragraph.\n\n## Chunking\n\nKeep these exact source lines.\n")
    saved = []
    backend = LanceDBBackend.__new__(LanceDBBackend)
    backend.embedding_fn = None
    backend.distance = "cosine"
    backend.table_name = "test"
    def create_table(_name, data, mode):
        saved.extend(record for batch in data for record in batch.to_pylist())
        return SimpleNamespace(create_fts_index=lambda *_a, **_kw: None)
    backend.db = SimpleNamespace(create_table=create_table)
    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    pipeline.config = {"document_paths": [str(source)], "chunk_size": 700, "chunk_overlap": 100}
    pipeline.vector_db = backend
    pipeline.bm25_index = None
    pipeline._save_database = lambda _docs: None
    pipeline._index_documents()
    passage = next(doc for doc in saved if doc["heading"] == "Book > Chunking")
    assert passage["file_path"] == str(source)
    assert passage["location"] == "L5-L7"
    assert (passage["start_line"], passage["end_line"]) == (5, 7)
    assert passage["original_content"] == "\n".join(source.read_text().splitlines()[4:7])


def test_chunk_settings_and_old_manifest_invalidate_index(tmp_path, monkeypatch):
    document = tmp_path / "chapter.txt"
    document.write_text("Source text.")
    monkeypatch.setenv("RAG_MCP_DATA_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("RAG_MCP_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("RAG_MCP_EMBED_PROVIDER", "none")
    monkeypatch.setattr(fetch, "_rag_system_v2_instances", {})
    configs = []
    def make_pipeline(*, config):
        configs.append(config)
        return SimpleNamespace(config=config)
    monkeypatch.setattr(fetch, "RAGPipelineV2", make_pipeline)
    first = fetch.get_rag_pipeline_v2(str(document))
    assert fetch.get_rag_pipeline_v2(str(document)) is first
    monkeypatch.setitem(fetch.RAG_CONFIG_V2, "chunk_size", 900)
    second = fetch.get_rag_pipeline_v2(str(document))
    assert second is not first and second.config["force_reindex"]
    from pathlib import Path
    manifest = Path(second.config["persist_dir"]) / "source-manifest.json"
    old = json.loads(manifest.read_text())
    old["version"] = 1
    old.pop("chunking", None)
    manifest.write_text(json.dumps(old))
    fetch.reset_pipelines()
    assert fetch.get_rag_pipeline_v2(str(document)).config["force_reindex"]


def test_repeated_passages_get_their_own_source_location():
    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    source = "same words\n\nsame words"
    first, cursor = pipeline._locate_chunk(source, "same words", ".txt", 0)
    second, _ = pipeline._locate_chunk(source, "same words", ".txt", cursor)
    assert (first, second) == ("L1", "L3")


def test_ingestion_uses_actual_offsets_for_repeated_long_source(tmp_path):
    source = tmp_path / "repeated.txt"
    source.write_text("same words\n" * 150)
    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    pipeline.config = {"chunk_size": 100, "chunk_overlap": 20}
    documents = build_documents(pipeline, [str(source)])
    assert len(documents) > 2
    assert documents[1]["start_line"] > documents[0]["start_line"]
    assert documents[2]["start_line"] > documents[1]["start_line"]
