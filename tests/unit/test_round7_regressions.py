import io
import json
import os
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest

from rag import cli_agents, server
from rag.cli import main
from rag.workspaces import WorkspaceStore


def test_workspace_type_filter_reaches_cli_and_mcp(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_MCP_WORKSPACE_DIR', str(tmp_path / 'workspaces'))
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(tmp_path / 'traces'))
    store = WorkspaceStore()
    store.replace('book', 'Book', [
        {'id': 'md', 'name': 'facts.md', 'sourceType': 'markdown', 'content': 'alpha'},
        {'id': 'py', 'name': 'default_store_smoke.py', 'sourceType': 'python', 'content': 'alpha private'},
    ])
    calls = []
    monkeypatch.setattr(server.importlib, 'import_module', lambda _: types.SimpleNamespace(
        fetchExternalKnowledge=lambda **kw: calls.append(kw) or 'answer'))
    output = io.StringIO()
    assert main(['query', 'alpha', '--workspace', 'book', '--type', 'md', '--format', 'json'],
                stdout=output, stderr=io.StringIO(), stdin=io.StringIO(''), environ=dict(os.environ)) == 0
    replies = []
    monkeypatch.setattr(server, '_result', lambda request_id, result: replies.append(result))
    monkeypatch.setattr(server, '_initialized', True)
    server._handle_request({'id': 1, 'method': 'tools/call', 'params': {
        'name': 'query_knowledge_base', 'arguments': {
            'query': 'alpha', 'workspace_id': 'book', 'extensions': ['md']}}})
    assert replies[-1]['content'][0]['text'] == 'answer'
    assert len(calls) == 2
    for call in calls:
        assert len(call['doc_path']) == 1
        assert Path(call['doc_path'][0]).suffix == '.md'
        assert not call['scope'].endswith(':all')


def test_rewritten_facets_preserve_caching():
    from rag.query_decomposition import decompose_query
    query = 'How do we measure retrieval latency? Recommend a practical caching strategy.'
    facets = decompose_query(query)
    assert any('caching' in facet for facet in facets)
    assert not any('golden dataset' in facet for facet in facets)


def test_lifecycle_facets_use_only_spare_capacity():
    from rag.query_decomposition import decompose_query
    query = 'How should we evaluate RAG? Describe the data.'
    facets = decompose_query(query, max_facets=3)
    assert facets[:2] == ['How should we evaluate RAG?', 'Describe the data.']
    assert 'ingestion' in facets[2]
    facets = decompose_query(query + ' Explain deployment. Compare costs.', max_facets=2)
    assert len(facets) == 2
    assert all(term in ' '.join(facets) for term in ['RAG', 'data', 'deployment', 'costs'])
    assert 'ingestion' not in ' '.join(facets)


def test_folder_skips_missing_office_parser_and_explicit_file_reports_it(tmp_path, monkeypatch):
    from rag import fetch
    from rag.bm25 import BM25Index
    from rag.parsing import extract_content
    monkeypatch.setitem(sys.modules, 'unstructured', None)
    monkeypatch.setenv('RAG_MCP_DATA_DIR', str(tmp_path / 'index'))
    monkeypatch.setenv('RAG_MCP_EMBED_PROVIDER', 'none')
    monkeypatch.setenv('RAG_MCP_RERANKER_TYPE', 'off')
    corpus = tmp_path / 'docs'
    corpus.mkdir()
    (corpus / 'readme.md').write_text('What changed? The release shipped.')
    (corpus / 'data.csv').write_text('change\nrelease')
    (corpus / 'report.docx').write_bytes(b'office fixture')
    def retrieve(**kwargs):
        index = BM25Index()
        for path in kwargs['doc_path']:
            index.add_document({'content': extract_content(path), 'file_path': path})
        return [document for document, _ in index.search(kwargs['query'], k=5)]
    monkeypatch.setattr(fetch, 'fetchExternalKnowledge', retrieve)
    from rag import openrouter_pipeline
    monkeypatch.setattr(openrouter_pipeline, 'synthesize_answer', lambda query, values, config: values[0]['content'])
    result = json.loads(server._query_knowledge_base('What changed?', str(corpus)))
    assert 'release shipped' in result['result']
    assert all(term in result['result'] for term in ('data.csv', 'report.docx', 'ragmcp[office]'))
    for name in ('data.csv', 'report.docx'):
        explicit = json.loads(server._query_knowledge_base('What changed?', str(corpus / name)))
        assert name in explicit['error'] and 'ragmcp[office]' in explicit['error']


@pytest.mark.parametrize('name', ['test_cli.py', 'worker_test.py', 'migrations/001.py'])
def test_explicit_noise_file_is_searchable(tmp_path, monkeypatch, name):
    from rag import fetch
    source = tmp_path / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('This test covers the release.')
    monkeypatch.setattr(fetch, 'fetchExternalKnowledge', lambda **kw: 'This test covers the release.')
    result = json.loads(server._query_knowledge_base('What does this test cover?', str(source)))
    assert result['result'] == 'This test covers the release.'
    if name.startswith('migrations/'):
        assert str(source) not in server._validate_rag_scope(tmp_path).files
    else:
        assert str(source) in server._validate_rag_scope(tmp_path).files


def test_claude_reconnect_removes_private_backup_after_success(tmp_path, monkeypatch):
    config = tmp_path / '.claude.json'
    old = {'type': 'stdio', 'command': 'old-server', 'args': ['mcp']}
    config.write_text(json.dumps({'secret': 'private-token', 'mcpServers': {'rag': old}}))
    monkeypatch.setattr(cli_agents.shutil, 'which', lambda *a, **k: 'claude')
    monkeypatch.setattr(cli_agents, 'server_command', lambda _: ['new-server', 'mcp'])
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        if 'remove' in args:
            backup, = tmp_path.glob('..claude.json.rag-backup-*')
            assert stat.S_IMODE(backup.stat().st_mode) == 0o600
            assert json.loads(backup.read_text()) == old
            return subprocess.CompletedProcess(args, 0, '', '')
        if 'add-json' in args and len([c for c in calls if 'add-json' in c]) > 1:
            return subprocess.CompletedProcess(args, 0, '', '')
        return subprocess.CompletedProcess(args, 1, '', '')
    monkeypatch.setattr(cli_agents.subprocess, 'run', run)
    assert cli_agents.connect_claude({'HOME': str(tmp_path)}, io.StringIO(), io.StringIO()) == 0
    assert not list(tmp_path.glob('..claude.json.rag-backup-*'))
