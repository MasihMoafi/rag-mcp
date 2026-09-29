import io
import json

import pytest

from rag import tracing
from rag.trace_reader import list_traces, read_trace
from rag.cli import main


@pytest.fixture
def traces(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', str(tmp_path))
    monkeypatch.delenv('RAG_MCP_TRACE_CONTENT', raising=False)
    with tracing.request_trace():
        with tracing.span('retrieval.lexical', candidate_limit=50):
            tracing.passages('candidates', [{'id': 'private', 'content': 'private text', 'score': 0.4}])
        with tracing.span('synthesis', model='test-model'):
            tracing.usage({'prompt_tokens': 12, 'completion_tokens': 3, 'cost': 0.002})
    return tmp_path


def test_reader_uses_real_trace_and_keeps_rank_usage_and_parentage(traces):
    listing = list_traces()
    summary, = listing['traces']
    assert summary['status'] == 'success'
    assert summary['span_count'] == 3
    assert summary['reported_cost_subtotal_usd'] == 0.002
    trace = read_trace(summary['trace_id'])
    assert trace['spans'][1]['parent_id'] == trace['spans'][0]['span_id']
    assert trace['spans'][1]['attributes']['candidates'][0]['rank'] == 1
    assert trace['spans'][2]['attributes']['usage']['input_tokens'] == 12
    assert 'private' not in json.dumps(trace)


def test_reader_never_follows_symlinks_or_reads_evaluations(traces):
    path, = traces.glob('trace-*.json')
    data = json.loads(path.read_text())
    data['capture_content'] = True
    data['spans'][0]['attributes']['query'] = 'secret query'
    target = traces / 'evaluations' / path.name
    target.parent.mkdir()
    target.write_text(json.dumps(data))
    path.unlink()
    path.symlink_to(target)
    assert list_traces()['traces'] == []
    assert read_trace(data['trace_id']) is None
    assert read_trace('../evaluations/' + data['trace_id']) is None
    path.unlink()
    path.write_text(json.dumps(data))
    assert read_trace(data['trace_id']) is None


def test_reader_drops_unknown_fields_and_handles_corrupt_records(traces):
    path, = traces.glob('trace-*.json')
    data = json.loads(path.read_text())
    data['apiKey'] = 'secret'
    data['spans'][0]['attributes']['request'] = {'api_key': 'secret'}
    data['spans'][1]['attributes']['candidates'][0]['content'] = 'secret'
    path.write_text(json.dumps(data))
    assert 'secret' not in json.dumps(read_trace(data['trace_id']))
    path.write_text('incomplete')
    assert list_traces()['skipped'] == 1
    assert list_traces()['traces'] == []


def test_disabled_missing_and_invalid_limits(traces, monkeypatch):
    assert read_trace('a' * 32) is None
    with pytest.raises(ValueError):
        list_traces(limit=101)
    monkeypatch.setenv('RAG_MCP_TRACE_DIR', 'off')
    assert list_traces()['enabled'] is False
    assert list_traces()['traces'] == []


def test_cli_lists_and_inspects_same_trace_without_running_query(traces):
    output = io.StringIO()
    assert main(['runs', '--json'], stdout=output) == 0
    trace_id = json.loads(output.getvalue())['traces'][0]['trace_id']
    output = io.StringIO()
    assert main(['runs', trace_id, '--json'], stdout=output) == 0
    assert json.loads(output.getvalue()) == read_trace(trace_id)
    output = io.StringIO()
    assert main(['runs', trace_id], stdout=output) == 0
    assert 'test-model' in output.getvalue()
    assert 'candidates' not in output.getvalue()
    output = io.StringIO()
    assert main(['runs'], stdout=output) == 0
    assert trace_id in output.getvalue()
    assert 'success' in output.getvalue()
    assert main(['runs', 'a' * 32], stderr=io.StringIO()) == 2


def test_cli_explicitly_opens_private_answer_and_evidence(traces):
    from rag.run_results import RunResultStore
    listed = io.StringIO()
    assert main(['runs', '--json'], stdout=listed) == 0
    trace_id = json.loads(listed.getvalue())['traces'][0]['trace_id']
    result = {'schema_version': 1, 'run_id': trace_id, 'status': 'success', 'answer': 'Friday.',
              'evidence': [{'rank': 1, 'name': 'facts.txt', 'content': 'Meeting Friday.'}]}
    RunResultStore().save(result)
    output = io.StringIO()
    assert main(['runs', trace_id, '--result', '--json'], stdout=output) == 0
    assert json.loads(output.getvalue()) == result
    output = io.StringIO()
    assert main(['runs', trace_id, '--result'], stdout=output) == 0
    assert 'Friday.' in output.getvalue() and 'Meeting Friday.' in output.getvalue()
    assert main(['runs', '--result'], stderr=io.StringIO()) == 2


def test_reader_paginates_latest_records_and_reports_failures(traces):
    import os
    first, = traces.glob('trace-*.json')
    from time import time
    os.utime(first, (time() - 60, time() - 60))
    try:
        with tracing.request_trace():
            with tracing.span('synthesis'):
                raise RuntimeError('private provider body')
    except RuntimeError:
        pass
    page = list_traces(limit=1)
    assert page['next_offset'] == 1
    assert page['traces'][0]['status'] == 'error'
    trace = read_trace(page['traces'][0]['trace_id'])
    assert trace['spans'][1]['error_class'] == 'RuntimeError'
    assert 'private provider' not in json.dumps(trace)
    next_page = list_traces(limit=1, offset=page['next_offset'])
    assert next_page['traces'][0]['trace_id'] == first.stem[6:]
    assert next_page['next_offset'] is None


def test_reader_refuses_nonregular_oversized_and_malformed_files(traces):
    import os
    path, = traces.glob('trace-*.json')
    trace_id = path.stem[6:]
    path.unlink()
    os.mkfifo(path)
    assert read_trace(trace_id) is None
    path.unlink()
    with path.open('wb') as stream:
        stream.truncate(4 * 1024 * 1024 + 1)
    assert read_trace(trace_id) is None
    path.write_text(json.dumps({'schema_version': 1, 'trace_id': trace_id,
                                'capture_content': False, 'spans': [None]}))
    assert read_trace(trace_id) is None


def test_menu_offers_run_inspection():
    from rag.cli_menu import MENU
    assert ("Inspect recent runs", ["runs"]) in MENU
