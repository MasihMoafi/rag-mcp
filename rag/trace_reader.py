"""Read the engine's completed metadata traces for both CLI and paired Studio."""
import json
import math
import os
from pathlib import Path
import re
import stat

from rag import tracing

_ID = re.compile(r'[0-9a-f]{32}')
_MAX_READ_BYTES = 4 * 1024 * 1024
_SCALARS = {
    'provider', 'model', 'embedding_model', 'reranker_model', 'cache_state',
    'backend', 'distance_metric', 'score_kind',
    'chunk_size', 'chunk_overlap', 'candidate_limit', 'final_limit',
    'source_count', 'input_count', 'output_count', 'facet_count',
    'candidates_count', 'selected_count', 'synthesis_evidence_count',
    'retry_count', 'timeout_seconds', 'http_status', 'output_token_cap',
    'skipped_reason', 'is_error', 'server_source_sha256_at_startup',
    'manifest_id', 'generation_id', 'prompt_sha256', 'source_id',
    'client', 'workspace_id', 'workspace_revision', 'configuration_revision', 'mode',
    'runtime_manifest_sha256', 'runtime_manifest_status',
}


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _attributes(raw):
    result = {key: value for key, value in raw.items() if key in _SCALARS
              and (value is None or type(value) in (str, bool) or _number(value) is not None)}
    usage = raw.get('usage')
    if isinstance(usage, dict):
        result['usage'] = {key: _number(usage.get(key)) for key in (
            'input_tokens', 'output_tokens', 'cost_usd', 'search_units', 'cached_tokens')}
    for key in ('candidates', 'selected', 'synthesis_evidence'):
        if isinstance(raw.get(key), list):
            result[key] = []
            for row in raw[key]:
                if not isinstance(row, dict) or not re.fullmatch(r'[0-9a-f]{64}', str(row.get('id', ''))):
                    continue
                item = {'id': row['id'], 'rank': _number(row.get('rank'))}
                score = row.get('score')
                if type(score) in (int, float) and math.isfinite(score):
                    item['score'] = score
                result[key].append(item)
    return result


def read_trace(trace_id, *, directory=None):
    if not isinstance(trace_id, str) or not _ID.fullmatch(trace_id):
        return None
    directory = Path(directory) if directory is not None else tracing._directory()
    if directory is None:
        return None
    path = directory / f'trace-{trace_id}.json'
    try:
        # Completed records are atomically published. Refuse symlinks and bound reads.
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            return None
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
        with os.fdopen(os.open(path, flags), 'rb') as stream:
            info = os.fstat(stream.fileno())
            if ((info.st_dev, info.st_ino) != (before.st_dev, before.st_ino)
                    or not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_READ_BYTES):
                return None
            record = json.loads(stream.read(_MAX_READ_BYTES + 1))
        if (not isinstance(record, dict) or record.get('schema_version') != 1
                or record.get('trace_id') != trace_id or record.get('capture_content') is not False
                or not isinstance(record.get('spans'), list) or not record['spans']):
            return None
        spans = []
        for raw in record['spans']:
            if not isinstance(raw, dict) or not isinstance(raw.get('attributes'), dict):
                return None
            if (not _ID.fullmatch(str(raw.get('span_id', '')))
                    or raw.get('status') not in ('success', 'error', 'cancelled', 'skipped')
                    or not isinstance(raw.get('operation'), str)
                    or _number(raw.get('duration_ms')) is None
                    or _number(raw.get('started_unix')) is None
                    or (raw.get('parent_id') is not None and not _ID.fullmatch(str(raw['parent_id'])))):
                return None
            span = {key: raw[key] for key in ('span_id', 'parent_id', 'operation', 'status', 'duration_ms', 'started_unix')}
            if isinstance(raw.get('error_class'), str):
                span['error_class'] = raw['error_class']
            span['attributes'] = _attributes(raw['attributes'])
            spans.append(span)
        accounting = record.get('accounting')
        accounting = accounting if isinstance(accounting, dict) else {}
        return {'schema_version': 1, 'trace_id': trace_id, 'spans': spans,
                'accounting': {key: _number(accounting.get(key)) for key in (
                    'reported_cost_subtotal_usd', 'unreported_cost_spans', 'estimated_cost_usd')}}
    except (OSError, ValueError, RecursionError, KeyError):
        return None


def list_traces(*, limit=20, offset=0, directory=None):
    if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or not 0 <= offset <= 10000:
        raise ValueError('limit must be 1–100 and offset 0–10000')
    directory = Path(directory) if directory is not None else tracing._directory()
    response = {'enabled': directory is not None, 'traces': [], 'skipped': 0, 'next_offset': None}
    if directory is None or not directory.exists():
        return response
    paths = []
    for path in directory.glob('trace-*.json'):
        if not re.fullmatch(r'trace-[0-9a-f]{32}\.json', path.name):
            continue
        try:
            paths.append((path.lstat().st_mtime_ns, path.name))
        except FileNotFoundError:
            continue  # Retention may remove a record while listing.
    paths.sort(reverse=True)
    for _, name in paths[offset:offset + limit]:
        trace = read_trace(name[6:-5], directory=directory)
        if trace is None:
            response['skipped'] += 1
            continue
        root = trace['spans'][0]
        response['traces'].append({
            'trace_id': trace['trace_id'], 'started_unix': root['started_unix'],
            'duration_ms': root['duration_ms'], 'status': root['status'],
            'span_count': len(trace['spans']), **trace['accounting'],
        })
    if len(paths) > offset + limit:
        response['next_offset'] = offset + limit
    return response
