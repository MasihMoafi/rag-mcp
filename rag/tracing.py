"""Bounded, local request traces. Content is opt-in; failures never replace answers."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import sys
from threading import Lock
import time
import uuid

from filelock import FileLock

_trace = ContextVar('rag_trace', default=None)
_parent = ContextVar('rag_span', default=None)
_passage_observer = ContextVar('rag_passage_observer', default=None)
_cancellation = ContextVar('rag_cancellation', default=None)
_health_lock = Lock()
_dropped = 0
_MAX_BYTES = 100 * 1024 * 1024
_MAX_AGE = 7 * 24 * 3600


class RunCancelled(BaseException):
    """Cooperative cancellation; must not become a provider failure or fallback."""


@contextmanager
def execution_context(observer, cancelled=None):
    observation_token = _passage_observer.set(observer)
    cancellation_token = _cancellation.set(cancelled)
    try:
        yield
    finally:
        _passage_observer.reset(observation_token)
        _cancellation.reset(cancellation_token)


def checkpoint():
    cancelled = _cancellation.get()
    if cancelled is not None and cancelled.is_set():
        raise RunCancelled()


def current_id():
    trace = _trace.get()
    return trace['trace_id'] if trace else None


def fail_current():
    current = _parent.get()
    if current is not None:
        current['status'] = 'error'


def _directory():
    configured = os.environ.get('RAG_MCP_TRACE_DIR', '').strip()
    if configured == 'off':
        return None
    return Path(configured).expanduser() if configured else Path(
        os.environ.get('XDG_STATE_HOME') or Path.home() / '.local/state'
    ) / 'ragmcp/traces'


def health():
    with _health_lock:
        return {'enabled': _directory() is not None, 'dropped_traces': _dropped,
                'counter_scope': 'process', 'content_capture': os.environ.get('RAG_MCP_TRACE_CONTENT') == '1'}


def annotate(**attributes):
    current = _parent.get()
    if current is not None:
        current['attributes'].update(attributes)
        if 'skipped_reason' in attributes:
            current['status'] = 'skipped'


def content(name, value):
    trace = _trace.get()
    if trace is not None and trace['capture_content']:
        annotate(**{name: value})


def identity(value):
    trace = _trace.get()
    if trace is None:
        return None
    return hmac.new(trace['_key'], str(value).encode(), hashlib.sha256).hexdigest()


def passages(name, documents):
    observer = _passage_observer.get()
    if observer is not None:
        observer(name, documents)
    trace = _trace.get()
    if trace is None:
        return
    rows = []
    for rank, document in enumerate(documents, 1):
        key = json.dumps([document.get('file_path', document.get('source')), document.get('id'),
                          document.get('chunk_id'), document.get('content')], ensure_ascii=False)
        row = {'id': identity(key), 'rank': rank}
        score = document.get('score')
        if type(score) in (int, float) and math.isfinite(score):
            row['score'] = score
        if trace['capture_content']:
            for field in ('content', 'original_content', 'id', 'chunk_id', 'file_path', 'source', 'location'):
                if field in document:
                    row['source_id' if field == 'id' else field] = document[field]
        rows.append(row)
    annotate(**{name: rows, name + '_count': len(rows)})


def usage(value):
    value = value if isinstance(value, dict) else {}
    def number(*names, integer=False):
        for name in names:
            item = value.get(name)
            if type(item) in ((int,) if integer else (int, float)) and item >= 0 and math.isfinite(item):
                return item
        return None
    cached = value.get('prompt_tokens_details')
    cached = cached.get('cached_tokens') if isinstance(cached, dict) else None
    annotate(usage={
        'input_tokens': number('prompt_tokens', 'input_tokens', integer=True),
        'output_tokens': number('completion_tokens', 'output_tokens', integer=True),
        'cost_usd': number('cost'), 'search_units': number('search_units', integer=True),
        'cached_tokens': cached if type(cached) is int and cached >= 0 else None,
        'source': 'provider' if value else 'unreported',
    })


@contextmanager
def span(operation, **attributes):
    trace = _trace.get()
    if trace is None:
        checkpoint()
        yield
        checkpoint()
        return
    parent = _parent.get()
    record = {'span_id': uuid.uuid4().hex, 'parent_id': parent['span_id'] if parent else None,
              'operation': operation, 'started_unix': time.time(),
              'status': 'skipped' if 'skipped_reason' in attributes else 'success',
              'attributes': attributes}
    trace['spans'].append(record)
    token = _parent.set(record)
    started = time.monotonic()
    try:
        checkpoint()
        yield
        checkpoint()
    except BaseException as error:
        record['status'] = 'error' if isinstance(error, Exception) else 'cancelled'
        # Exception class only, never exception text/provider bodies.
        record['error_class'] = type(error).__name__
        raise
    finally:
        record['duration_ms'] = (time.monotonic() - started) * 1000
        _parent.reset(token)


def traced(operation):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with span(operation):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def _persist(directory, record):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    data = json.dumps(record, ensure_ascii=False, allow_nan=False).encode()
    if len(data) > _MAX_BYTES:
        raise OSError('Trace exceeds retention budget')
    # A single writer protects retention and publication across processes. No
    # shared append stream, and incomplete files never look like complete traces.
    with FileLock(directory / '.write.lock', timeout=1):
        files = sorted((p for p in directory.glob('trace-*.json')
                        if re.fullmatch(r'trace-[0-9a-f]{32}\.json', p.name)), key=lambda p: p.stat().st_mtime)
        sizes = {p: p.stat().st_size for p in files}
        total = sum(sizes.values())
        cutoff = time.time() - _MAX_AGE
        if record['capture_content']:
            if total + len(data) > _MAX_BYTES:
                raise OSError('Evaluation capture budget full; archive explicitly')
        else:
            for path in files:
                if path.stat().st_mtime < cutoff or total + len(data) > _MAX_BYTES:
                    path.unlink()
                    total -= sizes[path]
        target = directory / f"trace-{record['trace_id']}.json"
        temporary = target.with_suffix('.tmp')
        try:
            with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def request_trace(trace_id=None):
    global _dropped
    directory = _directory()
    if directory is None or _trace.get() is not None:
        yield
        return
    if trace_id is not None and not re.fullmatch(r'[0-9a-f]{32}', trace_id):
        raise ValueError('Invalid run id')
    record = {'schema_version': 1, 'trace_id': trace_id or uuid.uuid4().hex,
              'capture_content': os.environ.get('RAG_MCP_TRACE_CONTENT') == '1',
              'identity_scope': 'request', 'spans': [], '_key': secrets.token_bytes(32)}
    if record['capture_content']:
        directory = directory / 'evaluations'
    token = _trace.set(record)
    try:
        with span('query'):
            yield
    finally:
        _trace.reset(token)
        record.pop('_key')
        usage_records = [s['attributes']['usage'] for s in record['spans'] if 'usage' in s['attributes']]
        reported = [u['cost_usd'] for u in usage_records if u['cost_usd'] is not None]
        record['accounting'] = {
            'reported_cost_subtotal_usd': sum(reported) if reported else None,
            'unreported_cost_spans': len(usage_records) - len(reported),
            'estimated_cost_usd': None,
        }
        try:
            _persist(directory, record)
        except Exception:
            with _health_lock:
                _dropped += 1
            print('[RAG] Local trace dropped; diagnostics degraded.', file=sys.stderr)


def trace_query(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with request_trace():
            result = function(*args, **kwargs)
            from .answer_contract import decode_public_answer
            with span('public.boundary'):
                _, failed = decode_public_answer(result)
                annotate(is_error=failed)
            current = _parent.get()
            if current is not None and failed:
                current['status'] = 'error'
            content('public_result', result)
            return result
    return wrapped
