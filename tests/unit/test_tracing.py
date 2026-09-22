"""Private diagnostics must describe the real operation without changing it."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from rag import tracing


@pytest.fixture
def trace_dir(tmp_path, monkeypatch):
    path = tmp_path / 'traces'
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(path))
    monkeypatch.delenv('RAG_MCP_TRACE_CONTENT', raising=False)
    return path


def read_traces(path):
    return [json.loads(p.read_text()) for p in path.rglob('trace-*.json')]


def test_trace_keeps_structure_usage_and_private_identity(trace_dir):
    passage = {'id': '/private-source::0', 'content': 'private-evidence', 'score': 0.8}
    with tracing.request_trace():
        with tracing.span('retrieve'):
            tracing.passages('selected', [passage])
            with tracing.span('provider'):
                tracing.usage({'prompt_tokens': 12, 'completion_tokens': 3, 'cost': 0.002})
        with tracing.span('synthesis'):
            tracing.passages('selected', [passage])
    record, = read_traces(trace_dir)
    assert record['schema_version'] == 1
    spans = {s['operation']: s for s in record['spans']}
    assert spans['provider']['parent_id'] == spans['retrieve']['span_id']
    assert spans['provider']['attributes']['usage']['input_tokens'] == 12
    assert spans['provider']['attributes']['usage']['cost_usd'] == 0.002
    assert spans['retrieve']['attributes']['selected'] == spans['synthesis']['attributes']['selected']
    assert all(s['duration_ms'] >= 0 for s in record['spans'])
    assert 'private-' not in json.dumps(record)
    assert next(trace_dir.glob('trace-*.json')).stat().st_mode & 0o777 == 0o600
    assert trace_dir.stat().st_mode & 0o777 == 0o700


def test_missing_or_invalid_usage_is_unknown(trace_dir):
    with tracing.request_trace():
        tracing.usage({'prompt_tokens': True, 'completion_tokens': -1, 'cost': float('nan')})
    record, = read_traces(trace_dir)
    usage = record['spans'][0]['attributes']['usage']
    assert usage['input_tokens'] is None
    assert usage['output_tokens'] is None
    assert usage['cost_usd'] is None


@pytest.mark.parametrize('error,status', [(RuntimeError('private-error'), 'error'), (KeyboardInterrupt(), 'cancelled')])
def test_failure_and_cancellation_preserve_exception(trace_dir, error, status):
    with pytest.raises(type(error)) as raised:
        with tracing.request_trace():
            with tracing.span('provider'):
                raise error
    assert raised.value is error
    record, = read_traces(trace_dir)
    assert all(s['status'] == status for s in record['spans'])
    assert 'private-error' not in json.dumps(record)


def test_parallel_requests_are_isolated(trace_dir):
    def run(number):
        with tracing.request_trace():
            with tracing.span('worker', number=number):
                pass
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(12)))
    records = read_traces(trace_dir)
    assert len(records) == 12
    assert len({r['trace_id'] for r in records}) == 12
    assert sorted(next(s['attributes']['number'] for s in r['spans'] if s['operation'] == 'worker') for r in records) == list(range(12))
    assert all(len(r['spans']) == 2 for r in records)


def test_disk_failure_does_not_replace_answer(trace_dir, monkeypatch, capsys):
    before = tracing.health()['dropped_traces']
    def fail(*args, **kwargs):
        raise OSError('private-disk-error')
    monkeypatch.setattr(tracing, '_persist', fail)
    with tracing.request_trace():
        answer = 'normal answer'
    assert answer == 'normal answer'
    assert tracing.health()['dropped_traces'] == before + 1
    assert 'private-disk-error' not in capsys.readouterr().err


def test_content_capture_requires_explicit_opt_in(trace_dir, monkeypatch):
    monkeypatch.setenv('RAG_MCP_TRACE_CONTENT', '1')
    with tracing.request_trace():
        tracing.content('query', 'private-query')
        tracing.passages('selected', [{'id': 'original-id', 'content': 'private-evidence'}])
    record, = read_traces(trace_dir)
    attributes = record['spans'][0]['attributes']
    assert attributes['query'] == 'private-query'
    assert attributes['selected'][0]['content'] == 'private-evidence'


def test_disabled_tracing_creates_no_files(trace_dir, monkeypatch):
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', 'off')
    with tracing.request_trace():
        with tracing.span('retrieve'):
            tracing.passages('selected', [{'id': 'a', 'content': 'text'}])
    assert not trace_dir.exists()


def test_process_writers_and_restart_do_not_lose_records(trace_dir):
    command = [sys.executable, '-c',
               'from rag.tracing import request_trace; '\
               '\nwith request_trace(): pass']
    writers = [subprocess.Popen(command) for _ in range(4)]
    assert all(writer.wait(timeout=15) == 0 for writer in writers)
    assert len(read_traces(trace_dir)) == 4
    subprocess.run(command, check=True, timeout=15)
    assert len(read_traces(trace_dir)) == 5


def test_retention_only_removes_owned_operational_traces(trace_dir, monkeypatch):
    trace_dir.mkdir()
    expired = trace_dir / ('trace-' + '0' * 32 + '.json')
    expired.write_text('{}')
    os.utime(expired, (1, 1))
    other = trace_dir / 'user-evidence.json'
    other.write_text('keep')
    monkeypatch.setattr(tracing, '_MAX_BYTES', 1500)
    for _ in range(5):
        with tracing.request_trace():
            pass
    assert not expired.exists()
    assert other.read_text() == 'keep'
    files = list(trace_dir.glob('trace-*.json'))
    assert 0 < len(files) < 5
    assert sum(p.stat().st_size for p in files) <= 1500


def test_full_evaluation_capture_is_preserved_not_rotated(trace_dir, monkeypatch):
    monkeypatch.setenv('RAG_MCP_TRACE_CONTENT', '1')
    with tracing.request_trace():
        tracing.content('query', 'old evidence')
    old, = (trace_dir / 'evaluations').glob('trace-*.json')
    original = old.read_bytes()
    monkeypatch.setattr(tracing, '_MAX_BYTES', len(original) + 1)
    with tracing.request_trace():
        tracing.content('query', 'new evidence')
    assert old.read_bytes() == original
    assert len(list((trace_dir / 'evaluations').glob('trace-*.json'))) == 1
