"""Budget refusal is tested without authenticating or calling inference."""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('diagnostic', Path(__file__).resolve().parents[2] / 'evals/run_diagnostic.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


@pytest.fixture
def key_data():
    return dict(limit=.03, limit_remaining=.03, usage=0, limit_reset=None, include_byok_in_limit=True)


def test_fresh_capped_key_covers_frozen_estimate(key_data):
    diagnostic.verify_budget(key_data, '.03', '.02229014')


@pytest.mark.parametrize('field,value', [('limit', None), ('limit', 1), ('limit', float('nan')),
    ('limit_remaining', .0004), ('usage', .001), ('limit_reset', 'monthly'),
    ('include_byok_in_limit', False), ('limit_remaining', True)])
def test_unknown_unlimited_resetting_or_spent_keys_are_refused(key_data, field, value):
    key_data[field] = value
    with pytest.raises(ValueError):
        diagnostic.verify_budget(key_data, '.03', '.02229014')


def test_old_allowance_cannot_fund_new_run(key_data):
    with pytest.raises(ValueError):
        diagnostic.verify_budget(key_data, '.0004', '.02229014')


def test_insufficient_budget_stops_before_authentication(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['diagnostic', str(tmp_path / 'source.txt'), '--budget-usd', '.0004', '--execute'])
    monkeypatch.setattr(diagnostic.subprocess, 'check_output', lambda *a, **k: json.dumps({'planning_estimate_usd': .02229014}))
    monkeypatch.setattr(diagnostic.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('No network permitted'))
    with pytest.raises(ValueError, match='insufficient'):
        diagnostic.main()


@pytest.mark.parametrize('fail', [False, True])
def test_runner_isolates_state_and_stops_on_first_error(key_data, monkeypatch, tmp_path, fail):
    from rag import fetch, settings
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['diagnostic', str(tmp_path / 'source.txt'), '--budget-usd', '.03',
                                    '--execute', '--output', str(output)])
    monkeypatch.setenv('OPENROUTER_API_KEY', 'private-test-key')
    monkeypatch.setenv('RAG_MCP_DATA_DIR', 'original-index')
    monkeypatch.setattr(diagnostic.subprocess, 'check_output', lambda *a, **k: json.dumps({'planning_estimate_usd': .02229014}))
    monkeypatch.setattr(diagnostic, 'datetime', SimpleNamespace(now=lambda *a: SimpleNamespace(
        date=lambda: SimpleNamespace(isoformat=lambda: '2026-09-22'))))
    monkeypatch.setattr(diagnostic.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(json.dumps({'data': key_data}).encode()))
    monkeypatch.setattr(fetch, 'reset_pipelines', lambda: None)
    calls = []
    def query(question, source):
        calls.append(question)
        assert settings.embedding_config()['embed_provider'] == 'openai_compatible'
        assert settings.embedding_config()['embed_api_key'] == 'private-test-key'
        assert os.environ['RAG_MCP_DATA_DIR'] == str(output / 'index')
        return json.dumps({'error': 'controlled failure'} if fail else {'result': 'controlled answer'})
    monkeypatch.setitem(sys.modules, 'server', SimpleNamespace(query_knowledge_base=query))
    assert diagnostic.main() == (1 if fail else 0)
    assert len(calls) == (1 if fail else 3)
    assert os.environ['RAG_MCP_DATA_DIR'] == 'original-index'
    raw = (output / 'answers.json').read_text()
    assert 'private-test-key' not in raw
    assert len(json.loads(raw)['cases']) == len(calls)
