"""The paid-run core with fakes for every paid call: no network, no key. Each guard is shown firing and passing."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('paidrun', Path(__file__).resolve().parents[2] / 'evals/paidrun.py')
assert spec is not None and spec.loader is not None
paidrun = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paidrun)

SETS = {'a': [1, 2, 3, 4], 'b': [5, 6, 7]}


def good(_name, item):
    return {'passages': [item]}


def make(tmp_path, results=good, answers=('yes',), unit=0.002, cap=1.0, smoke=2, usage=None):
    calls, answers = [], iter(answers)
    def call(name, item):
        calls.append((name, item))
        return results(name, item)
    run = paidrun.Run(SETS, call, paidrun.has_passages, unit, cap, tmp_path / 'results.jsonl', smoke=smoke, usage=usage,
                      confirm=lambda _prompt: next(answers))
    return run, calls


def test_smoke_stops_at_the_first_unproven_output_before_the_full_run(tmp_path):
    run, calls = make(tmp_path, lambda n, i: {'code': 'engine_busy'} if (n, i) == ('b', 5) else good(n, i))
    with pytest.raises(SystemExit, match=r'first failure \(b #1\): \{"code": "engine_busy"\}'):
        run.start()
    assert calls == [('a', 1), ('a', 2), ('b', 5)]  # valid JSON without passages stops it; nothing after


def test_smoke_covers_every_set_prints_outputs_and_the_rest_runs_after_yes(tmp_path, capsys):
    run, calls = make(tmp_path)
    assert run.start() == 7
    out = capsys.readouterr().out
    assert 'smoke a #1: {"passages": [1]}' in out and 'smoke b #1: {"passages": [5]}' in out
    assert 'estimate: 4 probes cost $0.0080, $0.0020 per item x 3 remaining = $0.0060' in out
    assert calls[:4] == [('a', 1), ('a', 2), ('b', 5), ('b', 6)]


def test_without_yes_nothing_beyond_the_smoke_is_spent(tmp_path):
    run, calls = make(tmp_path, answers=('no',))
    with pytest.raises(SystemExit, match='not confirmed'):
        run.start()
    assert len(calls) == 4


def test_estimate_is_the_probe_times_n_not_a_recorded_mean(tmp_path, capsys):
    spent = iter([0.0, 0.26])  # the key's usage before and after the 4 probes: $0.065 each, far above the $0.002 unit
    run, _ = make(tmp_path, answers=('no',), usage=lambda: next(spent))
    with pytest.raises(SystemExit):
        run.start()
    assert '4 probes cost $0.2600, $0.0650 per item x 3 remaining = $0.1950' in capsys.readouterr().out


def test_cap_stops_before_the_call_that_passes_it_by_own_count(tmp_path):
    run, calls = make(tmp_path, unit=0.002, cap=0.01, smoke=1)
    with pytest.raises(SystemExit, match=r'5 calls made, one more passes the \$0.01 cap'):
        run.start()
    assert len(calls) == 5


def test_a_parsed_record_without_the_proving_field_is_redone(tmp_path):
    (tmp_path / 'results.jsonl').write_text(json.dumps({'collection': 'a', 'n': 1, 'result': {'code': 'engine_busy'}}) + '\n'
                                            + json.dumps({'collection': 'a', 'n': 2, 'result': {'passages': ['x']}}) + '\n')
    run, calls = make(tmp_path, smoke=0)
    assert run.done() == {('a', 2)}
    run.start()
    assert calls[0] == ('a', 1) and ('a', 2) not in calls


def test_cli_search_waits_out_engine_busy_and_returns_an_unproven_error(monkeypatch):
    outputs = iter(['{"code": "engine_busy"}', 'not json at all'])
    monkeypatch.setattr(paidrun.time, 'sleep', lambda _s: None)
    monkeypatch.setattr(paidrun.subprocess, 'run', lambda *a, **_k: subprocess.CompletedProcess(a, 1, next(outputs), 'ValueError: x'))
    result = paidrun.cli_search('python', '.', {}, 'w', 'q')
    assert result == {'error': 'ValueError: x', 'code': 1} and not paidrun.has_passages(result)


def environ(tmp_path):
    return {'HOME': str(tmp_path / 'home'), 'PATH': os.environ['PATH']}


def test_real_stores_are_rag_mcp_defaults(tmp_path, monkeypatch):
    env = environ(tmp_path)
    monkeypatch.setenv('HOME', env['HOME'])
    for var in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME', *paidrun.STORES):
        monkeypatch.delenv(var, raising=False)
    from rag import index_config, settings, tracing, workspaces
    assert paidrun.real_stores(env) == {settings.config_path(env).parent, Path(index_config.index_root_path()),
                                        workspaces._default_root(env), tracing._directory()}


@pytest.mark.parametrize('inside', ['.config/rag', '.local/share/rag/index/x', '.local/state/ragmcp'])
def test_isolation_refuses_the_real_store(tmp_path, inside):
    with pytest.raises(SystemExit, match='is the real store'):
        paidrun.isolated_env(tmp_path / 'home' / inside, environ(tmp_path))


def test_isolation_refuses_a_git_ignored_folder_of_the_repo(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    (repo / '.gitignore').write_text('store/\n')
    with pytest.raises(SystemExit, match='git-ignored folder of the repo'):
        paidrun.isolated_env(repo / 'store' / 'run', environ(tmp_path), repo=repo)


def test_isolation_accepts_a_cache_store_and_locks_each_store_once(tmp_path):
    run = tmp_path / 'home/.cache/rag-experiments/run-1'
    env = paidrun.isolated_env(run, environ(tmp_path))
    assert {k: env[k] for k in paidrun.STORES} == {'RAG_MCP_CONFIG': str(run / 'config.json'), 'RAG_MCP_DATA_DIR': str(run / 'index'),
                                                   'RAG_MCP_WORKSPACE_DIR': str(run / 'workspaces'), 'RAG_MCP_TRACE_DIR': str(run / 'traces')}
    assert env['PATH'] == os.environ['PATH']
    with pytest.raises(SystemExit, match='another run holds this store'):
        paidrun.isolated_env(run, environ(tmp_path))
    paidrun.isolated_env(run.with_name('run-2'), environ(tmp_path))
