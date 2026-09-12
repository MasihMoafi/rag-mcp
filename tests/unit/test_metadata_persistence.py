"""Metadata publication must not expose incomplete JSON."""

import json
from types import SimpleNamespace

import pytest

from rag import ingestion


def test_failed_metadata_serialization_preserves_previous_file(tmp_path, monkeypatch):
    pipeline = SimpleNamespace(config={'persist_dir': str(tmp_path)})
    path = tmp_path / 'docs.json'
    previous = [{'content': 'previous evidence'}]
    path.write_text(json.dumps(previous))
    def interrupted(value, handle, **kwargs):
        handle.write('[partial')
        raise OSError('simulated write failure')
    monkeypatch.setattr(ingestion.json, 'dump', interrupted)
    with pytest.raises(OSError, match='simulated write failure'):
        ingestion.save_documents(pipeline, [{'content': 'new evidence'}])
    assert json.loads(path.read_text()) == previous
    assert sorted(p.name for p in tmp_path.iterdir()) == ['docs.json']


def test_metadata_save_replaces_complete_file(tmp_path):
    pipeline = SimpleNamespace(config={'persist_dir': str(tmp_path)})
    documents = [{'content': 'متن', 'location': 'p.1-2'}]
    ingestion.save_documents(pipeline, documents)
    assert json.loads((tmp_path / 'docs.json').read_text()) == documents


def test_failed_publication_preserves_previous_file_and_cleans_temporary(tmp_path, monkeypatch):
    from utils import persistence
    target = tmp_path / 'source-manifest.json'
    target.write_text('{"version": 1}')
    def failed_replace(*args):
        raise OSError('rename failed')
    monkeypatch.setattr(persistence.os, 'replace', failed_replace)
    with pytest.raises(OSError, match='rename failed'):
        persistence.write_json_atomic(target, {'version': 2})
    assert json.loads(target.read_text()) == {'version': 1}
    assert list(tmp_path.iterdir()) == [target]
