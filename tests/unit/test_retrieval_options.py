"""The three retrieval options tested on EnterpriseRAG-Bench (Nov 2026)."""

from rag.index_config import build_source_stamp
from rag.ingestion import source_title
from rag.lancedb_backend import LanceDBBackend
from rag.reranking import balanced_candidates


def test_title_comes_from_the_file_name_but_never_from_a_hash():
    assert source_title("/c/ab12/rotate-api-keys_for prod.txt") == "rotate api keys for prod"
    assert source_title("/c/" + "3f" * 32 + ".txt") == ""


def test_each_search_keeps_its_share_of_the_candidates():
    docs = {name: {"id": name} for name in "abcdef"}
    vector = [(docs[n], 0) for n in "abcd"]
    keyword = [(docs[n], 0) for n in "ef"]
    fused = [docs[n] for n in "abcdef"]  # a plain cut of four would drop both keyword hits
    assert [d["id"] for d in balanced_candidates(fused, [vector, keyword], 4)] == list("abef")


def test_stemmed_keyword_search_finds_another_form_of_the_word(tmp_path):
    def hits(stemming):
        backend = LanceDBBackend(f"t{stemming}", None, persist_path=str(tmp_path / str(stemming)),
                                 keyword_stemming=stemming)
        backend.add_documents_batch([{"id": "1", "content": "The deployment failed twice."},
                                     {"id": "2", "content": "Lunch is at noon."}])
        return [hit["id"] for hit in backend.search_fts("deploy", top_k=5)]

    assert hits(False) == [] and hits(True) == ["1"]


def test_options_change_the_index_stamp_only_when_on():
    config = {"backend": "lancedb", "distance_metric": "cosine", "chunk_size": 700,
              "chunk_overlap": 100, "document_paths": [], "use_contextual": False}
    plain = build_source_stamp(config, {}, {})
    assert "options" not in plain
    assert build_source_stamp(dict(config, keyword_stemming=True), {}, {})["options"] == {
        "keyword_stemming": True}
