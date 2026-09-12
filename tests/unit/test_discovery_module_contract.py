from pathlib import Path
from types import SimpleNamespace

import pytest

from rag import fetch


def test_path_shortcuts_environment_and_empty_input(monkeypatch, tmp_path):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('RAG_TEST_SOURCE', str(tmp_path / 'source'))
    monkeypatch.chdir(tmp_path)
    assert fetch.resolve_path(' Desktop ') == str(tmp_path / 'Desktop')
    assert fetch.resolve_path('$RAG_TEST_SOURCE/file.md') == str(tmp_path / 'source' / 'file.md')
    assert fetch.resolve_path('relative.md') == str(tmp_path / 'relative.md')
    assert fetch.resolve_path('') is None


def test_legacy_query_keeps_patchable_pipeline_and_discovery(monkeypatch, tmp_path):
    selected = [str(tmp_path / 'source.md')]
    result = [{'content': 'evidence'}]
    calls = []
    def discover(directory, **kwargs):
        calls.append(('discover', directory, kwargs['max_files']))
        return selected
    def pipeline(doc_path):
        calls.append(('pipeline', doc_path))
        return SimpleNamespace(search=lambda query: result)
    monkeypatch.setattr(fetch, 'find_all_indexable_files', discover)
    monkeypatch.setattr(fetch, 'get_rag_pipeline_v2', pipeline)
    assert fetch.fetchExternalKnowledge('question', str(tmp_path)) is result
    assert calls == [('discover', str(tmp_path), 501), ('pipeline', selected)]
    replacement = [{'content': 'replacement'}]
    monkeypatch.setattr(fetch, 'fetchExternalKnowledgeV2', lambda *args, **kwargs: replacement)
    assert fetch.fetchExternalKnowledge('question') is replacement
    assert len(calls) == 2


@pytest.mark.parametrize('query', ['', None, 42])
def test_query_validation_remains_before_pipeline(monkeypatch, query):
    def unexpected(**kwargs):
        pytest.fail('invalid query must not construct a pipeline')
    monkeypatch.setattr(fetch, 'get_rag_pipeline_v2', unexpected)
    with pytest.raises(ValueError, match='Enter a question'):
        fetch.fetchExternalKnowledge(query)


def test_discovery_module_reexports_and_shared_os(monkeypatch, tmp_path):
    from rag import discovery
    assert fetch.resolve_path is discovery.resolve_path
    assert fetch.find_all_indexable_files is discovery.find_all_indexable_files
    assert fetch.find_relevant_files is discovery.find_relevant_files
    assert fetch.os is discovery.os
    assert fetch.query_documents is fetch.fetchExternalKnowledgeV2
    file = tmp_path / 'source.md'
    file.touch()
    monkeypatch.setattr(fetch.os, 'listdir', lambda path: [file.name])
    assert discovery.find_all_indexable_files(str(tmp_path)) == [str(file)]


@pytest.mark.parametrize('count', [500, 501])
def test_folder_limit_does_not_silently_drop_documents(monkeypatch, tmp_path, count):
    for index in range(count):
        (tmp_path / f'doc-{index}.txt').touch()
    selected = []
    def pipeline(doc_path):
        selected.extend(doc_path)
        return SimpleNamespace(search=lambda query: [])
    monkeypatch.setattr(fetch, 'get_rag_pipeline_v2', pipeline)
    if count == 501:
        with pytest.raises(ValueError, match='500-document limit'):
            fetch.query_documents('question', str(tmp_path))
        assert not selected
    else:
        assert fetch.query_documents('question', str(tmp_path)) == []
        assert len(selected) == 500
