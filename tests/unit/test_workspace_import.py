import io
import json
import contextlib
from http.client import HTTPConnection
from threading import Thread

import pytest

from rag.cli import main
from rag.workspaces import WorkspaceConflict, WorkspaceStore


def test_cli_source_import_json_keeps_parser_diagnostics_off_stdout(tmp_path, monkeypatch):
    from rag import workspace_import
    def noisy_import(*_args, **_kwargs):
        print('parser diagnostic')
        return {'schema_version': 1, 'workspace_id': 'book', 'revision': 'revision',
                'document_count': 1, 'total_bytes': 10, 'imported_count': 1, 'source_ids': ['source']}
    monkeypatch.setattr(workspace_import, 'import_workspace_sources', noisy_import)
    output, diagnostics = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(output):
        assert main(['sources', 'add', 'book', '--path', str(tmp_path / 'chapter.md'), '--json'],
                    stdout=output, stderr=diagnostics,
                    environ={'RAG_MCP_WORKSPACE_DIR': str(tmp_path / 'workspaces')}) == 0
    assert json.loads(output.getvalue())['revision'] == 'revision'
    assert 'parser diagnostic' in diagnostics.getvalue()


def test_cli_folder_import_is_shared_and_reimport_updates_in_place(tmp_path, monkeypatch):
    folder = tmp_path / 'book'
    folder.mkdir()
    (folder / 'one.md').write_text('First chapter.')
    (folder / 'two.md').write_text('Second chapter.')
    (folder / 'saved.html').write_text('<p>Duplicate saved page.</p>')
    root = tmp_path / 'workspaces'
    store = WorkspaceStore(root)
    store.replace('book', 'Book', [{'id': 'browser-source', 'name': 'Browser source',
                                    'sourceType': 'text', 'content': 'Keep this Studio source.'}])
    monkeypatch.setattr('rag.cli.user_settings.run_setup', lambda **kwargs: pytest.fail('Model setup started'))
    output = io.StringIO()
    assert main(['sources', 'add', 'book', '--path', str(folder), '--type', 'md', '--json'],
                stdout=output, stderr=io.StringIO(),
                environ={'RAG_MCP_WORKSPACE_DIR': str(root)}) == 0
    result = json.loads(output.getvalue())
    assert result['imported_count'] == 2 and result['document_count'] == 3
    assert 'First chapter.' not in output.getvalue()
    snapshot = store.snapshot('book')
    assert [item['content'] for item in snapshot['documents']] == [
        'Keep this Studio source.', 'First chapter.', 'Second chapter.']
    assert {item['sourceType'] for item in snapshot['documents'][1:]} == {'markdown'}

    from rag import server
    from rag.companion import create_companion_server
    monkeypatch.setenv('RAG_MCP_WORKSPACE_DIR', str(root))
    replies = []
    monkeypatch.setattr(server, '_result', lambda request_id, result: replies.append(result))
    monkeypatch.setattr(server, '_initialized', False)
    server._handle_request({'id': 0, 'method': 'initialize'})
    server._handle_request({'id': 1, 'method': 'tools/call',
                            'params': {'name': 'get_workspaces', 'arguments': {'workspace_id': 'book'}}})
    assert json.loads(replies[-1]['content'][0]['text'])['document_count'] == 3
    gateway = create_companion_server(token='fixture-token', workspace_store=store,
                                      codex_ready=lambda: False, metrics_path=tmp_path / 'metrics.json')
    worker = Thread(target=gateway.serve_forever, daemon=True)
    worker.start()
    connection = HTTPConnection('127.0.0.1', gateway.server_port, timeout=5)
    try:
        connection.request('GET', '/v1/workspaces/book/snapshot', headers={
            'Authorization': 'Bearer fixture-token', 'Origin': 'http://localhost:3000'})
        response = connection.getresponse()
        assert response.status == 200
        remote = json.loads(response.read())
        assert [(item['id'], item['content']) for item in remote['documents']] == [
            (item['id'], item['content']) for item in snapshot['documents']]
        connection.request('GET', '/v1/workspaces/book', headers={
            'Authorization': 'Bearer fixture-token', 'Origin': 'http://localhost:3000'})
        metadata = connection.getresponse()
        assert metadata.status == 200 and 'First chapter.' not in metadata.read().decode()
    finally:
        connection.close()
        gateway.shutdown()
        gateway.server_close()
        worker.join(timeout=2)

    (folder / 'one.md').write_text('Revised first chapter.')
    output = io.StringIO()
    assert main(['sources', 'add', 'book', '--path', str(folder), '--type', 'md', '--json'],
                stdout=output, stderr=io.StringIO(),
                environ={'RAG_MCP_WORKSPACE_DIR': str(root)}) == 0
    revised = store.snapshot('book')
    assert [item['id'] for item in revised['documents']] == [item['id'] for item in snapshot['documents']]
    assert revised['documents'][1]['content'] == 'Revised first chapter.'
    assert revised['documents'][0]['content'] == 'Keep this Studio source.'


def test_import_rejects_concurrent_workspace_change(tmp_path, monkeypatch):
    from rag.workspace_import import import_workspace_sources

    source = tmp_path / 'chapter.md'
    source.write_text('Chapter.')
    store = WorkspaceStore(tmp_path / 'workspaces')
    initial = store.replace('book', 'Book', [{'id': 'first', 'name': 'First',
                                             'sourceType': 'text', 'content': 'Original.'}])
    original_replace = store.replace

    def concurrent_replace(workspace_id, name, documents, *, expected_revision=None):
        original_replace('book', 'Book', [{'id': 'concurrent', 'name': 'Studio update',
                                          'sourceType': 'text', 'content': 'Latest.'}],
                         expected_revision=initial['revision'])
        return original_replace(workspace_id, name, documents, expected_revision=expected_revision)

    monkeypatch.setattr(store, 'replace', concurrent_replace)
    with pytest.raises(WorkspaceConflict):
        import_workspace_sources(store, 'book', source)
    assert [item['id'] for item in store.snapshot('book')['documents']] == ['concurrent']


def test_unreadable_chapter_does_not_publish_a_partial_workspace(tmp_path, monkeypatch):
    from rag import workspace_import

    folder = tmp_path / 'book'
    folder.mkdir()
    (folder / 'one.md').write_text('Readable chapter.')
    (folder / 'two.md').write_text('Text the parser loses.')
    extract = workspace_import.extract_content
    monkeypatch.setattr(workspace_import, 'extract_content',
                        lambda file: '   ' if file.endswith('two.md') else extract(file))
    store = WorkspaceStore(tmp_path / 'workspaces')
    before = store.replace('book', 'Book', [{'id': 'browser-source', 'name': 'Studio source',
                                           'sourceType': 'text', 'content': 'Keep me.'}])
    with pytest.raises(ValueError, match='No readable text'):
        workspace_import.import_workspace_sources(store, 'book', folder, ['md'])
    after = store.snapshot('book')
    assert after['revision'] == before['revision']
    assert [item['id'] for item in after['documents']] == ['browser-source']


def test_folder_import_skips_empty_files(tmp_path):
    from rag.workspace_import import import_workspace_sources

    folder = tmp_path / 'package'
    folder.mkdir()
    (folder / '__init__.py').write_text('')
    (folder / 'blank.py').write_text('\n  \n')
    (folder / 'core.py').write_text('def search(): pass\n')
    store = WorkspaceStore(tmp_path / 'workspaces')
    result = import_workspace_sources(store, 'code', folder, ['py'])
    assert result['imported_count'] == 1 and result['skipped_empty'] == ['__init__.py', 'blank.py']
    assert [item['name'] for item in store.snapshot('code')['documents']] == ['core.py']
    with pytest.raises(ValueError, match='No readable text'):
        import_workspace_sources(store, 'code', folder / '__init__.py')
    (folder / 'core.py').write_text('')
    with pytest.raises(ValueError, match='No readable text'):
        import_workspace_sources(store, 'code', folder, ['py'])


def test_mcp_import_requires_current_revision_and_returns_metadata_only(tmp_path, monkeypatch):
    from rag import server
    from rag import workspace_import

    root = tmp_path / 'workspaces'
    monkeypatch.setenv('RAG_MCP_WORKSPACE_DIR', str(root))
    source = tmp_path / 'chapter.md'
    source.write_text('Private source text.')
    replies = []
    monkeypatch.setattr(server, '_result', lambda request_id, result: replies.append(result))
    monkeypatch.setattr(server, '_initialized', False)
    server._handle_request({'id': 0, 'method': 'initialize'})
    server._handle_request({'id': 1, 'method': 'tools/list'})
    assert 'add_workspace_sources' in {item['name'] for item in replies[-1]['tools']}

    def add(revision):
        server._handle_request({'id': 2, 'method': 'tools/call', 'params': {
            'name': 'add_workspace_sources', 'arguments': {
                'workspace_id': 'book', 'path': str(source), 'revision': revision}}})
        return replies[-1]

    first = add(None)
    assert first['isError'] is False
    value = json.loads(first['content'][0]['text'])
    assert value['document_count'] == value['imported_count'] == 1
    assert 'Private source text.' not in first['content'][0]['text']
    source.write_text('Revised private text.')
    extractor = workspace_import.extract_content
    monkeypatch.setattr(workspace_import, 'extract_content', lambda path: pytest.fail('Stale call parsed source'))
    stale = add(None)
    monkeypatch.setattr(workspace_import, 'extract_content', extractor)
    assert stale['isError'] is True
    assert WorkspaceStore(root).snapshot('book')['documents'][0]['content'] == 'Private source text.'
    fresh = add(value['revision'])
    assert fresh['isError'] is False
    assert WorkspaceStore(root).snapshot('book')['documents'][0]['content'] == 'Revised private text.'
