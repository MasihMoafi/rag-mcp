from pathlib import Path
from types import SimpleNamespace

import pytest

from rag import fetch


EXTENSIONS = (
    '.rs', '.toml', '.yaml', '.yml', '.py', '.md', '.txt', '.json',
    '.js', '.ts', '.tsx', '.jsx', '.c', '.h', '.cpp', '.hpp',
    '.go', '.sh', '.bash', '.zsh', '.css', '.html', '.sql', '.java',
    '.kt', '.proto', '.pdf', '.ipynb', '.rst', '.ini', '.cfg', '.conf',
    '.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp', '.docx', '.pptx', '.xlsx', '.csv',
)


@pytest.mark.parametrize('setting,value', [('backend', 'qdrant'), ('distance_metric', 'dot'),
                                          ('rrf_k', 12), ('bm25_k1', 2.0), ('bm25_b', 0.5)])
def test_pipeline_cache_honors_engine_settings(monkeypatch, tmp_path, setting, value):
    source = tmp_path / 'source.txt'
    source.write_text('Cedar')
    monkeypatch.setenv('RAG_MCP_DATA_DIR', str(tmp_path / 'index'))
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('RAG_MCP_EMBED_PROVIDER', 'none')
    monkeypatch.setattr(fetch, '_rag_system_v2_instances', {})
    monkeypatch.setattr(fetch, 'RAGPipelineV2', lambda config: SimpleNamespace(config=config))
    first = fetch.get_rag_pipeline_v2(str(source))
    monkeypatch.setitem(fetch.RAG_CONFIG_V2, setting, value)
    second = fetch.get_rag_pipeline_v2(str(source))
    assert second is not first
    assert second.config[setting] == value
    assert second.config['force_reindex'] == (setting in {'backend', 'distance_metric'})
    assert fetch.get_rag_pipeline_v2(str(source)) is second


def ordered_files(monkeypatch, directory, names):
    for name in names:
        (directory / name).touch()
    original = fetch.os.listdir
    monkeypatch.setattr(fetch.os, 'listdir', lambda path: names if Path(path) == directory else original(path))
    return [str(directory / name) for name in names]


def test_discovery_preserves_supported_extensions(monkeypatch, tmp_path):
    names = ['source' + extension.upper() for extension in EXTENSIONS]
    expected = ordered_files(monkeypatch, tmp_path, names + ['source.tex', 'source.exe'])[:len(names)]
    assert fetch.find_all_indexable_files(str(tmp_path)) == expected
    assert fetch.find_relevant_files('', str(tmp_path), max_files=100) == expected
    assert fetch.find_relevant_files('source', str(tmp_path), max_files=100) == expected


@pytest.mark.parametrize('limit', [0, 1, 3, -1])
@pytest.mark.parametrize('query', ['', '   ', 'cedar', 'cedar cedar alpha'])
def test_filename_ranking_and_limits(monkeypatch, tmp_path, query, limit):
    paths = ordered_files(monkeypatch, tmp_path, ['cedar.py', 'cedar-alpha.md', '.cedar.txt', 'other.html', 'cedar.exe'])
    candidates = paths[:4]
    if query.strip():
        scores = [(sum(word in Path(path).name.lower() for word in query.split()), path) for path in candidates]
        candidates = [path for score, path in sorted(scores, key=lambda item: item[0], reverse=True) if score]
    assert fetch.find_relevant_files(query.upper(), str(tmp_path), limit) == candidates[:limit]


def test_recursive_filters_do_not_leak_into_filename_search(monkeypatch, tmp_path):
    names = ['.hidden.md', 'skip.py', 'keep.md', 'omit.md']
    paths = ordered_files(monkeypatch, tmp_path, names)
    (tmp_path / 'nested').mkdir()
    (tmp_path / 'nested' / 'deep.md').touch()
    names.append('nested')
    (tmp_path / 'node_modules').mkdir()
    (tmp_path / 'node_modules' / 'module.md').touch()
    names.append('node_modules')
    assert fetch.find_all_indexable_files(str(tmp_path), include_patterns=['*.md'],
        excluded_paths={paths[3]}, max_depth=0) == [paths[2]]
    assert fetch.find_all_indexable_files(str(tmp_path), exclude_patterns=['omit*'], max_files=1) == [paths[1]]
    assert fetch.find_relevant_files('', str(tmp_path), 10) == paths
    assert fetch.find_all_indexable_files(str(tmp_path / 'missing')) == []
    assert fetch.find_relevant_files('cedar', str(tmp_path / 'missing')) == []


@pytest.mark.parametrize('as_list', [False, True])
@pytest.mark.parametrize('setting', [None, 'contextual_model', 'contextual_max_tokens',
    'contextual_timeout', 'contextual_source_max_chars', 'contextual_start_chunks', 'contextual_prev_chunks'])
def test_scoped_pipeline_cache_and_manifest(monkeypatch, tmp_path, as_list, setting):
    source = tmp_path / 'source.md'
    source.write_text('Cedar', encoding='utf-8')
    root = tmp_path / 'indexes'
    monkeypatch.setenv('RAG_MCP_DATA_DIR', str(root))
    monkeypatch.setattr(fetch, '_rag_system_v2_instances', {})
    embedding = {'embed_provider': 'none', 'embed_model': ''}
    retrieval = dict(top_k=3, reranker_type='none', reranker_model='', contextual_model='test',
        contextual_max_tokens=10, contextual_timeout=2, contextual_source_max_chars=100,
        contextual_start_chunks=1, contextual_prev_chunks=1)
    monkeypatch.setattr(fetch, 'embedding_config', lambda: embedding)
    monkeypatch.setattr(fetch, 'retrieval_config', lambda: retrieval)
    monkeypatch.setattr(fetch, 'contextual_retrieval_enabled', lambda settings: False)
    instances = []
    def build(config):
        instance = SimpleNamespace(config=config)
        instances.append(instance)
        return instance
    monkeypatch.setattr(fetch, 'RAGPipelineV2', build)
    scope = [str(source)] if as_list else str(source)
    first = fetch.get_rag_pipeline_v2(scope)
    assert first.config['document_paths'] == [str(source)]
    assert first.config['persist_dir'] == str(root / fetch._safe_dir_name(str(source), embedding))
    assert first.config['force_reindex'] is True
    assert fetch.get_rag_pipeline_v2(scope) is first
    fetch.reset_pipelines()
    reopened = fetch.get_rag_pipeline_v2(scope)
    assert reopened.config['force_reindex'] is False
    source.write_text('Cedar changed', encoding='utf-8')
    refreshed = fetch.get_rag_pipeline_v2(scope)
    assert refreshed is not reopened
    assert refreshed.config['force_reindex'] is True
    assert len(instances) == 3
    if setting:
        retrieval[setting] = 'changed' if setting == 'contextual_model' else retrieval[setting] + 1
        changed = fetch.get_rag_pipeline_v2(scope)
        assert changed is not refreshed
        assert changed.config[setting] == retrieval[setting]
        assert changed.config['force_reindex'] is (setting != 'contextual_timeout')
        assert fetch.get_rag_pipeline_v2(scope) is changed


def test_directory_stamp_is_sorted_independent_of_enumeration(monkeypatch, tmp_path):
    source = tmp_path / 'sources'
    source.mkdir()
    paths = [source / 'b.md', source / 'a.md']
    for path in paths:
        path.write_text(path.name)
    monkeypatch.setenv('RAG_MCP_DATA_DIR', str(tmp_path / 'indexes'))
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('RAG_MCP_EMBED_PROVIDER', 'none')
    monkeypatch.setenv('RAG_MCP_CONTEXTUAL_RETRIEVAL', 'off')
    monkeypatch.setattr(fetch, '_rag_system_v2_instances', {})
    monkeypatch.setattr(fetch, 'RAGPipelineV2', lambda config: SimpleNamespace(config=config))
    monkeypatch.setattr(Path, 'rglob', lambda self, pattern: iter(paths))
    first = fetch.get_rag_pipeline_v2(str(source))
    expected = [[str(path), path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns]
                for path in sorted(paths)]
    assert first.config['source_stamp']['sources'] == expected
    paths.reverse()
    assert fetch.get_rag_pipeline_v2(str(source)) is first
