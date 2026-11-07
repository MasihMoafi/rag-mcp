import io
import getpass
import json
import os
from pathlib import Path

import httpx
import pytest

from rag import server
from rag.cli import main
from rag.workspaces import WorkspaceStore


def test_setup_key_inputs_and_warning(tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    env = {'RAG_MCP_CONFIG': str(config)}
    base = ['setup', '--provider', 'openai_compatible', '--no-verify']
    for options, supplied in [(['--api-key-stdin'], 'stdin-secret\n'), ([], '')]:
        monkeypatch.setattr(getpass, 'getpass', lambda *a, **kw: 'prompt-secret')
        stdin = io.StringIO(supplied)
        if not options:
            stdin.isatty = lambda: True
        assert main(base + options, stdin=stdin, stdout=io.StringIO(), stderr=io.StringIO(), environ=env) == 0
        assert json.loads(config.read_text())['embed_api_key'] == ('stdin-secret' if options else 'prompt-secret')
    assert main(base, stdin=io.StringIO(), stdout=io.StringIO(), stderr=io.StringIO(), environ={**env, 'RAG_MCP_EMBED_API_KEY': 'env-secret'}) == 0
    assert json.loads(config.read_text())['embed_api_key'] == 'env-secret'
    err = io.StringIO()
    assert main(base + ['--api-key', 'fake-key'], stdout=io.StringIO(), stderr=err, environ=env) == 0
    assert 'process' in err.getvalue() and 'history' in err.getvalue()


def test_provider_401_body_never_reaches_logs_or_trace(tmp_path, monkeypatch, capsys):
    import openai
    from rag import fetch, tracing
    from rag.embeddings import OpenAICompatibleWrapper
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(tmp_path / 'traces'))
    client = openai.OpenAI(api_key='fake-key', base_url='https://example.test/v1', max_retries=0,
                           http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401, json={'error': 'Bearer fake-key'}))))
    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: client)
    wrapper = OpenAICompatibleWrapper('model', 'fake-key', 'https://example.test/v1')
    monkeypatch.setattr(fetch, 'open_or_build', lambda *args, **kwargs: wrapper.embed_documents(['alpha']))
    monkeypatch.setenv('RAG_MCP_DATA_DIR', str(tmp_path / 'index'))
    source = tmp_path / 'source.txt'
    source.write_text('alpha')
    with tracing.request_trace():
        with pytest.raises(Exception):
            fetch.get_rag_pipeline_v2(str(source))
    output = capsys.readouterr()
    recorded = ''.join(p.read_text() for p in (tmp_path / 'traces').rglob('*.json'))
    assert 'fake-key' not in output.out + output.err + recorded
    monkeypatch.setenv('RAG_MCP_WORKSPACE_DIR', str(tmp_path / 'workspaces'))
    from rag.workspace_import import import_workspace_sources
    import_workspace_sources(WorkspaceStore(), 'W', source)
    out, err = io.StringIO(), io.StringIO()
    assert main(['search', 'alpha', '--workspace', 'W', '--json'], stdout=out, stderr=err,
                stdin=io.StringIO(), environ=dict(os.environ)) != 0
    output = capsys.readouterr()
    recorded = ''.join(p.read_text() for p in (tmp_path / 'traces').rglob('*.json'))
    assert 'fake-key' not in out.getvalue() + err.getvalue() + output.out + output.err + recorded


def test_workspace_original_extension_cli_and_mcp(tmp_path, monkeypatch):
    from rag.workspace_import import import_workspace_sources
    monkeypatch.setenv('RAG_MCP_WORKSPACE_DIR', str(tmp_path / 'workspaces'))
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(tmp_path / 'traces'))
    source = tmp_path / 'main.py'
    source.write_text('alpha python source')
    import_workspace_sources(WorkspaceStore(), 'W', source)
    calls = []
    from rag import fetch
    monkeypatch.setattr(fetch, 'fetchExternalKnowledge', lambda **kw: calls.append(kw) or 'alpha found')
    for extension, expected in [('py', True), ('txt', False)]:
        output, err = io.StringIO(), io.StringIO()
        code = main(['query', 'alpha', '--workspace', 'W', '--type', extension, '--format', 'json'], stdout=output, stderr=err, stdin=io.StringIO(), environ=dict(os.environ))
        assert (code == 0) is expected
        response = json.loads(server.query_knowledge_base('alpha', workspace_id='W', extensions=[extension]))
        assert ('result' in response) is expected
    assert len(calls) == 2
    assert all(len(call['doc_path']) == 1 for call in calls)


def test_instruction_content_retrieves_backup_schedule_bm25(tmp_path, monkeypatch):
    from rag import fetch, openrouter_pipeline
    import tantivy
    builder = tantivy.SchemaBuilder()
    builder.add_text_field('content', stored=True)
    index = tantivy.Index(builder.build())
    writer = index.writer()
    writer.add_document(tantivy.Document(content='backup schedule: nightly'))
    writer.commit()
    index.reload()
    searcher = index.searcher()
    def retrieve(**kwargs):
        query = index.parse_query(kwargs['query'], ['content'])
        return [{'content': 'backup schedule: nightly', 'file_path': 'facts.txt'} for _ in searcher.search(query, 5).hits]
    monkeypatch.setattr(fetch, 'fetchExternalKnowledge', retrieve)
    monkeypatch.setattr(openrouter_pipeline, 'synthesize_answer', lambda query, docs, config: docs[0]['content'] if docs else 'No relevant information found.')
    source = tmp_path / 'facts.txt'
    source.write_text('backup schedule: nightly')
    result = json.loads(server._query_knowledge_base('What is the retention period? For each system, give the backup schedule.', str(source)))
    assert 'backup schedule: nightly' in result['result']
    result = json.loads(server._query_knowledge_base('What is the retention period? Return the backup schedule as JSON.', str(source)))
    assert 'backup schedule: nightly' in result['result']
    result = json.loads(server._query_knowledge_base('What is the retention period? Give the backup schedule with citations.', str(source)))
    assert 'backup schedule: nightly' in result['result']
    result = json.loads(server._query_knowledge_base('What is the retention period? Give the backup schedule with citations and retention details.', str(source)))
    assert 'backup schedule: nightly' in result['result']


def test_noise_only_folder_is_searchable_but_mixed_folder_skips_noise(tmp_path, monkeypatch):
    from rag import fetch
    folder = tmp_path / 'tests'
    folder.mkdir()
    (folder / 'test_login.py').write_text('Login covers password checks.')
    calls = []
    monkeypatch.setattr(fetch, 'fetchExternalKnowledge', lambda **kw: calls.append(kw) or 'Login covers password checks.')
    result = json.loads(server._query_knowledge_base('What does login cover?', str(folder)))
    assert 'Login covers' in result['result']
    assert [Path(p).name for p in calls[-1]['doc_path']] == ['test_login.py']
    (folder / 'login.py').write_text('Login handles sign in.')
    result = json.loads(server._query_knowledge_base('What does login cover?', str(folder)))
    assert 'result' in result
    assert [Path(p).name for p in calls[-1]['doc_path']] == ['login.py']
    (folder / 'login.py').rename(folder / 'login.md')
    result = json.loads(server._query_knowledge_base('What does login cover?', str(folder), extensions=['py']))
    assert 'error' in result


