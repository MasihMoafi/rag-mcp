"""Durable retrieval batches shared by Studio, CLI, and MCP."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
import uuid
from threading import Event

from filelock import FileLock, Timeout

from rag.evaluation_sets import EvaluationSetStore
from rag.workspaces import WorkspaceConflict, WorkspaceStore
from utils.persistence import write_json_atomic

_ID = re.compile(r'[0-9a-f]{32}')
_MAX_BYTES = 2 * 1024 * 1024


class EvaluationRunStore:
    def __init__(self, workspaces: WorkspaceStore | None = None):
        self.workspaces = workspaces or WorkspaceStore()

    def path(self, workspace_id, batch_id):
        self.workspaces.validate_id(workspace_id)
        if not isinstance(batch_id, str) or not _ID.fullmatch(batch_id):
            raise ValueError('Invalid evaluation batch ID.')
        directory = self.workspaces._workspace(workspace_id) / 'evaluation-runs'
        if directory.is_symlink():
            raise OSError('Evaluation batch storage is invalid.')
        return directory / f'{batch_id}.json'

    @staticmethod
    def _validate(value, workspace_id, batch_id):
        if (not isinstance(value, dict) or value.get('schema_version') != 1
                or value.get('workspace_id') != workspace_id or value.get('batch_id') != batch_id
                or value.get('status') not in ('running', 'success', 'error', 'cancelled', 'interrupted')
                or not isinstance(value.get('questions'), list) or not value['questions']
                or not isinstance(value.get('rows'), list)
                or len(value['rows']) > len(value['questions'])):
            raise ValueError('Invalid evaluation batch.')
        if EvaluationSetStore._normalize(value['questions']) != value['questions']:
            raise ValueError('Invalid evaluation questions.')
        for field in ('workspace_revision', 'question_revision'):
            if not isinstance(value.get(field), str) or not 1 <= len(value[field]) <= 100:
                raise ValueError('Invalid evaluation revision.')
        if not isinstance(value.get('configuration_revision'), str) or not re.fullmatch(r'[0-9a-f]{64}', value['configuration_revision']):
            raise ValueError('Invalid configuration revision.')
        if (not isinstance(value.get('source_ids'), list) or not 1 <= len(value['source_ids']) <= 500
                or not all(isinstance(item, str) and 0 < len(item) <= 256 for item in value['source_ids'])
                or len(set(value['source_ids'])) != len(value['source_ids'])):
            raise ValueError('Invalid evaluation sources.')
        for field in ('started_unix', 'finished_unix'):
            number = value.get(field)
            if field == 'finished_unix' and number is None:
                continue
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError('Invalid evaluation time.')
        if value.get('error') is not None and (not isinstance(value['error'], str) or len(value['error']) > 4000):
            raise ValueError('Invalid evaluation error.')
        for question, row in zip(value['questions'], value['rows']):
            if (not isinstance(row, dict) or row.get('question_id') != question['id']
                    or not isinstance(row.get('run_id'), str) or not _ID.fullmatch(row['run_id'])
                    or row.get('status') not in ('success', 'error', 'cancelled')):
                raise ValueError('Invalid evaluation row.')
            for field in ('passage_count', 'source_count'):
                if type(row.get(field)) is not int or not 0 <= row[field] <= 10000:
                    raise ValueError('Invalid evaluation count.')
            duration = row.get('duration_ms')
            if (row['source_count'] > row['passage_count'] or type(duration) not in (int, float)
                    or not math.isfinite(duration) or duration < 0
                    or (row.get('error') is not None and (not isinstance(row['error'], str) or len(row['error']) > 4000))):
                raise ValueError('Invalid evaluation row metrics.')

    def get(self, workspace_id, batch_id):
        self.workspaces.get(workspace_id)
        path = self.path(workspace_id, batch_id)
        try:
            with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)), 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
                    raise OSError('Evaluation batch is invalid.')
                value = json.load(stream)
            self._validate(value, workspace_id, batch_id)
        except (ValueError, TypeError, UnicodeError, RecursionError) as error:
            raise OSError('Evaluation batch is invalid.') from error
        if value['status'] == 'running':
            try:
                with FileLock(str(path) + '.lock', timeout=0):
                    value = {**value, 'status': 'interrupted', 'error': 'Execution stopped before the batch finished. Completed results are preserved.'}
            except Timeout:
                pass
        return value

    def save(self, value):
        self._validate(value, value['workspace_id'], value['batch_id'])
        path = self.path(value['workspace_id'], value['batch_id'])
        path.parent.mkdir(mode=0o700, exist_ok=True)
        path.parent.chmod(0o700)
        if len(json.dumps(value, indent=2).encode()) > _MAX_BYTES:
            raise OSError('Evaluation batch exceeds storage limit.')
        write_json_atomic(path, value)
        path.chmod(0o600)

    def list(self, workspace_id, *, limit=20, offset=0):
        self.workspaces.get(workspace_id)
        if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or not 0 <= offset <= 10000:
            raise ValueError('Invalid batch list bounds.')
        directory = self.path(workspace_id, '0' * 32).parent
        paths = sorted((p for p in directory.glob('*.json') if _ID.fullmatch(p.stem)),
                       key=lambda p: p.lstat().st_mtime_ns, reverse=True)
        batches, skipped = [], 0
        for path in paths[offset:offset + limit]:
            try:
                value = self.get(workspace_id, path.stem)
                batches.append({key: value[key] for key in ('batch_id', 'status', 'started_unix', 'finished_unix', 'workspace_revision', 'question_revision', 'configuration_revision')} | {
                    'question_count': len(value['questions']), 'completed_count': len(value['rows']),
                    'passage_count': sum(row['passage_count'] for row in value['rows'])})
            except OSError:
                skipped += 1
        return {'schema_version': 1, 'workspace_id': workspace_id, 'batches': batches, 'skipped': skipped,
                'next_offset': offset + limit if len(paths) > offset + limit else None}


def execute_evaluation(workspace_id, *, batch_id=None, question_revision=None, workspace_revision=None,
                       source_ids=None, configuration_revision=None, client='cli', cancelled=None,
                       store=None, executor=None, result_store=None):
    from rag.control import read_config
    from rag.execution import execute_run

    store = store or EvaluationRunStore()
    batch_id = batch_id or uuid.uuid4().hex
    path = store.path(workspace_id, batch_id)

    def existing():
        try:
            previous = store.get(workspace_id, batch_id)
        except FileNotFoundError:
            return None
        if (question_revision is not None and previous['question_revision'] != question_revision
                or workspace_revision is not None and previous['workspace_revision'] != workspace_revision
                or source_ids is not None and previous['source_ids'] != sorted(source_ids)
                or configuration_revision is not None and previous['configuration_revision'] != configuration_revision):
            raise WorkspaceConflict('Batch ID belongs to a different evaluation. Start a new batch.')
        return previous

    previous = existing()
    if previous is not None:
        return previous
    question_set = EvaluationSetStore(store.workspaces).get(workspace_id)
    snapshot = store.workspaces.snapshot(workspace_id)
    if question_revision is not None and question_revision != question_set['revision']:
        raise WorkspaceConflict('Evaluation questions changed. Reload before running.')
    if workspace_revision is not None and workspace_revision != snapshot['revision']:
        raise WorkspaceConflict('Workspace sources changed. Sync before running.')
    selected = sorted(source_ids if source_ids is not None else [item['id'] for item in snapshot['documents']])
    available = {item['id'] for item in snapshot['documents']}
    if not selected or not set(selected).issubset(available) or len(set(selected)) != len(selected):
        raise ValueError('Select available sources before evaluation.')
    if not question_set['questions']:
        raise ValueError('Add evaluation questions before running.')
    revision = configuration_revision or hashlib.sha256(json.dumps(read_config(), sort_keys=True).encode()).hexdigest()
    cancelled = cancelled or Event()
    path.parent.mkdir(mode=0o700, exist_ok=True)
    with FileLock(store.workspaces.root / '.evaluation.lock', timeout=0), FileLock(str(path) + '.lock', timeout=0):
        previous = existing()
        if previous is not None:
            if previous['status'] == 'running':
                previous.update(status='interrupted', error='Execution stopped before the batch finished. Completed results are preserved.')
            return previous
        value = {'schema_version': 1, 'workspace_id': workspace_id, 'batch_id': batch_id, 'status': 'running',
                 'started_unix': time.time(), 'finished_unix': None, 'error': None,
                 'question_revision': question_set['revision'], 'workspace_revision': snapshot['revision'],
                 'configuration_revision': revision, 'source_ids': selected, 'questions': question_set['questions'], 'rows': []}
        store.save(value)
        try:
            for question in value['questions']:
                if cancelled.is_set():
                    value.update(status='cancelled', error='Evaluation cancelled.')
                    break
                started = time.monotonic()
                result = (executor or execute_run)(question['question'], workspace_id=workspace_id, client=client,
                    run_id=uuid.uuid4().hex, workspace_store=store.workspaces, workspace_revision=value['workspace_revision'],
                    source_ids=selected, configuration_revision=revision, mode='retrieve', cancelled=cancelled, result_store=result_store)
                evidence = result.get('evidence', []) if result['status'] == 'success' else []
                value['rows'].append({'question_id': question['id'], 'run_id': result['run_id'], 'status': result['status'],
                    'passage_count': len(evidence), 'source_count': len({item['source_id'] for item in evidence}),
                    'duration_ms': round((time.monotonic() - started) * 1000, 3), 'error': result.get('error')})
                store.save(value)
                if result['status'] != 'success' or result.get('persistence_error'):
                    value.update(status=result['status'] if result['status'] != 'success' else 'error',
                                 error=result.get('persistence_error') or result.get('error'))
                    break
            else:
                value['status'] = 'success'
            if cancelled.is_set() and value['status'] in {'running', 'success'}:
                value.update(status='cancelled', error='Evaluation cancelled.')
        except KeyboardInterrupt:
            cancelled.set()
            value.update(status='cancelled', error='Evaluation cancelled.')
        except Exception:
            value.update(status='error', error='Evaluation execution failed. Completed results are preserved.')
        finally:
            value['finished_unix'] = time.time()
            store.save(value)
        return value
