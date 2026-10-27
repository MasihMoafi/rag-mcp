"""A rebuilt index keeps only its own files; earlier versions do not accumulate on disk."""

from rag.lancedb_backend import LanceDBBackend
from tests.integration.test_incremental_index import RecordingEmbedder


def test_rebuild_does_not_retain_previous_versions(tmp_path):
    backend = LanceDBBackend('fixture', RecordingEmbedder(), vector_size=2, persist_path=str(tmp_path / 'db'))
    table = tmp_path / 'db' / 'fixture.lance'
    counts = []
    for edit in range(3):
        backend.add_documents_batch([{'id': '1', 'content': f'Passage one, edit {edit}.'}])
        counts.append((len(list((table / 'data').iterdir())), len(list((table / '_versions').iterdir()))))
    assert counts == [counts[0]] * 3, counts
    assert backend.table.count_rows() == 1
