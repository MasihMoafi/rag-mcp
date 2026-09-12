"""MCP setting delegation preserves order, validation and cache invalidation."""

from unittest.mock import Mock

import pytest

import server
from rag import cli, fetch


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
