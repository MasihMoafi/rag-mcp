"""MCP setting delegation preserves order, validation and cache invalidation."""

from unittest.mock import Mock

import pytest

import server
from rag import cli, fetch


def test_settings_report_uses_captured_server_identity(monkeypatch):
    from rag import detect
    assert server._SERVER_SOURCE_SHA256 == server.hashlib.sha256(server.Path(server.__file__).read_bytes()).hexdigest()
    monkeypatch.setattr(server, '_SERVER_SOURCE_SHA256', 'startup-fingerprint')
    monkeypatch.setattr(server.Path, 'read_bytes', lambda *args: pytest.fail('Do not rehash changed disk contents'))
    monkeypatch.setattr(server.user_settings, 'describe_settings', lambda: [])
    monkeypatch.setattr(server, 'embedding_config', lambda: {
        'embed_provider': 'none', 'embed_model': '', 'ollama_url': 'unused'})
    monkeypatch.setattr(server.user_settings, 'retrieval_config', lambda: {})
    monkeypatch.setattr(detect, 'detect_all', lambda *args: {})
    assert server._settings_report()['runtime'] == {
        'pid': server.os.getpid(), 'server_source_sha256_at_startup': 'startup-fingerprint'}


def test_mcp_retrieval_error_does_not_echo_provider_payload(monkeypatch, tmp_path):
    import json

    failure = RuntimeError('provider body contains sk-private-example and private document text')
    monkeypatch.setattr(server, '_validate_rag_scope', lambda path, extensions=None: None)
    monkeypatch.setattr(fetch, 'fetchExternalKnowledge', Mock(side_effect=failure))
    output = json.loads(server.query_knowledge_base('question', str(tmp_path)))
    assert 'error' in output
    assert 'RuntimeError' in output['error']
    assert 'sk-private-example' not in output['error']
    assert 'private document text' not in output['error']


def test_multi_part_question_retrieves_facets_but_synthesizes_once(monkeypatch, tmp_path):
    import json
    import types
    from rag import openrouter_pipeline

    document = tmp_path / 'chapter.md'
    document.write_text('Evaluation evidence.')
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs['query'])
        marker = len(calls)
        return [
            {'content': f'facet {marker} passage {rank}', 'file_path': str(document)}
            for rank in range(1, 4)
        ]

    captured = {}

    def synthesize(query, passages, config):
        captured.update(query=query, passages=passages)
        return 'Combined answer.'

    monkeypatch.setattr(
        server.importlib,
        'import_module',
        lambda _name: types.SimpleNamespace(fetchExternalKnowledge=fetch),
    )
    monkeypatch.setattr(openrouter_pipeline, 'synthesize_answer', synthesize)
    query = (
        'Why can retrieval metrics look good while the answer is poor? '
        'Explain the failure points, and give the practical evaluation approach.'
    )

    output = json.loads(server.query_knowledge_base(query, str(document)))

    assert output == {'result': 'Combined answer.'}
    assert calls == [
        'Why can retrieval metrics look good while the answer is poor?',
        'Explain the failure points',
        'give the practical evaluation approach.',
    ]
    assert captured['query'] == query
    assert len(captured['passages']) == 5
    assert {passage['content'].split()[1] for passage in captured['passages']} == {'1', '2', '3'}


@pytest.mark.parametrize('read_fails', [False, True])
@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_server_shutdown_releases_pipelines(monkeypatch, read_fails, cleanup_fails, capsys):
    failure = OSError('input failed')
    read = Mock(side_effect=failure) if read_fails else Mock(return_value=None)
    reset = Mock(side_effect=RuntimeError('cleanup failed') if cleanup_fails else None)
    monkeypatch.setattr(server, '_read_payload', read)
    monkeypatch.setattr(fetch, 'reset_pipelines', reset)
    if read_fails:
        with pytest.raises(OSError) as caught:
            server.main()
        assert caught.value is failure
    else:
        server.main()
    reset.assert_called_once_with()
    output = capsys.readouterr()
    assert not output.out
    if cleanup_fails:
        assert 'cleanup' in output.err.lower()


@pytest.fixture
def delegation(monkeypatch, tmp_path):
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    calls = []

    def configure(args, **kwargs):
        calls.append((args.setting, args.value, args.provider, args.no_verify, kwargs['verify']))
        return 0

    monkeypatch.setattr(cli, '_config_command', configure)
    monkeypatch.setattr(server.user_settings, 'describe_settings', lambda: ['configured'])
    reset = Mock()
    monkeypatch.setattr(fetch, 'reset_pipelines', reset)
    return calls, reset


@pytest.mark.parametrize('failure', ['validation', 'probe'])
def test_failed_request_preserves_live_settings(monkeypatch, tmp_path, failure):
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    path = server.user_settings.save_settings({'top_k': '5', 'custom': 'preserved'})
    original = path.read_bytes()
    reset = Mock()
    monkeypatch.setattr(fetch, 'reset_pipelines', reset)
    monkeypatch.setattr(cli, '_probe_embedding', Mock(side_effect=RuntimeError('probe failed')))
    arguments = {'top': 9, 'contextual_retrieval': 'invalid'} if failure == 'validation' else {
        'top': 9, 'embedding_provider': 'ollama', 'embedding_model': 'fixture'}
    assert 'error' in server._apply_settings(arguments)
    assert path.read_bytes() == original
    reset.assert_not_called()


def test_successful_request_publishes_all_settings(monkeypatch, tmp_path):
    monkeypatch.setenv('RAG_MCP_CONFIG', str(tmp_path / 'config.json'))
    server.user_settings.save_settings({'top_k': '5', 'custom': 'preserved'})
    reset = Mock()
    monkeypatch.setattr(fetch, 'reset_pipelines', reset)
    result = server._apply_settings({'top': 9, 'reranker': 'off'})
    assert result['changed'] == ['top', 'reranker']
    assert server.user_settings.load_settings() == {
        'top_k': '9', 'custom': 'preserved', 'reranker_type': 'off', 'reranker_model': ''}
    reset.assert_called_once_with()


def test_setting_order_and_alias_are_preserved(delegation):
    calls, reset = delegation
    result = server._apply_settings({'contextual_retrieval': 'on', 'reranker': 'off', 'top': 3})
    assert result == {'changed': ['top', 'reranker', 'contextual_retrieval'], 'settings': ['configured']}
    assert calls == [('top', '3', None, True, None),
                     ('reranker', 'off', None, True, None),
                     ('contextual', 'on', None, True, None)]
    reset.assert_called_once_with()


@pytest.mark.parametrize('failed', ['top', 'reranker', 'contextual'])
def test_validation_failure_stops_later_changes(monkeypatch, delegation, failed):
    calls, reset = delegation

    def configure(args, **kwargs):
        calls.append(args.setting)
        if args.setting == failed:
            kwargs['stderr'].write('invalid setting\n')
            return 2
        return 0

    monkeypatch.setattr(cli, '_config_command', configure)
    result = server._apply_settings({'top': 3, 'reranker': 'off', 'contextual_retrieval': 'on'})
    assert result == {'error': 'invalid setting'}
    order = ['top', 'reranker', 'contextual']
    assert calls == order[:order.index(failed) + 1]
    reset.assert_not_called()


def test_empty_change_does_not_reset(delegation):
    calls, reset = delegation
    assert 'error' in server._apply_settings({})
    assert calls == []
    reset.assert_not_called()


def test_embedding_keeps_probe_and_runs_after_retrieval(delegation):
    calls, reset = delegation
    result = server._apply_settings({'embedding_model': 'test', 'embedding_provider': 'ollama', 'top': 2})
    assert result['changed'] == ['top', 'embedding']
    assert calls == [('top', '2', None, True, None),
                     ('embedding', 'test', 'ollama', False, cli._probe_embedding)]
    reset.assert_called_once_with()
