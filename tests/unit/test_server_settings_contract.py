"""MCP setting delegation preserves order, validation and cache invalidation."""

from unittest.mock import Mock

import pytest

import server
from rag import cli, fetch


def test_mcp_retrieval_error_does_not_echo_provider_payload(monkeypatch, tmp_path):
    import json

    failure = RuntimeError('provider body contains sk-private-example and private document text')
    monkeypatch.setattr(server, '_validate_rag_scope', lambda path: None)
    monkeypatch.setattr(fetch, 'fetchExternalKnowledge', Mock(side_effect=failure))
    output = json.loads(server.query_knowledge_base('question', str(tmp_path)))
    assert 'error' in output
    assert 'RuntimeError' in output['error']
    assert 'sk-private-example' not in output['error']
    assert 'private document text' not in output['error']


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
def delegation(monkeypatch):
    calls = []

    def configure(args, **kwargs):
        calls.append((args.setting, args.value, args.provider, args.no_verify, kwargs['verify']))
        return 0

    monkeypatch.setattr(cli, '_config_command', configure)
    monkeypatch.setattr(server.user_settings, 'describe_settings', lambda: ['configured'])
    reset = Mock()
    monkeypatch.setattr(fetch, 'reset_pipelines', reset)
    return calls, reset


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
