"""Private human judgments about passages retained for one evaluation batch."""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat

from filelock import FileLock

from rag.evaluation_runs import EvaluationRunStore
from rag.run_results import RunResultStore
from rag.workspaces import WorkspaceConflict, WorkspaceStore
from utils.persistence import write_json_atomic

_HEX32 = re.compile(r'[0-9a-f]{32}')
_HEX64 = re.compile(r'[0-9a-f]{64}')
_MAX_BYTES = 2 * 1024 * 1024


class EvaluationLabelStore:
    def __init__(self, workspaces: WorkspaceStore | None = None, results: RunResultStore | None = None):
        self.batches = EvaluationRunStore(workspaces)
        self.results = results or RunResultStore()

    def path(self, workspace_id: str, batch_id: str):
        return self.batches.path(workspace_id, batch_id).with_suffix('.labels.json')

    @staticmethod
    def _validate(value, workspace_id, batch_id, batch):
        if (not isinstance(value, dict) or value.get('schema_version') != 1
                or value.get('workspace_id') != workspace_id or value.get('batch_id') != batch_id
                or not isinstance(value.get('revision'), str) or not _HEX32.fullmatch(value['revision'])
                or not isinstance(value.get('labels'), list) or len(value['labels']) > 20000):
            raise ValueError('Invalid evaluation labels.')
        runs = {row['run_id']: row for row in batch['rows']}
        seen = set()
        for item in value['labels']:
            if (not isinstance(item, dict) or set(item) != {'question_id', 'run_id', 'rank', 'source_id',
                    'start_offset', 'end_offset', 'content_sha256', 'label'}
                    or item.get('run_id') not in runs or item.get('question_id') != runs[item['run_id']]['question_id']
                    or type(item.get('rank')) is not int or not 1 <= item['rank'] <= runs[item['run_id']]['passage_count']
                    or item.get('source_id') not in batch['source_ids']
                    or type(item.get('start_offset')) is not int or item['start_offset'] < 0
                    or type(item.get('end_offset')) is not int or item['end_offset'] <= item['start_offset']
                    or not isinstance(item.get('content_sha256'), str) or not _HEX64.fullmatch(item['content_sha256'])
                    or item.get('label') not in ('relevant', 'irrelevant')
                    or (item['run_id'], item['rank']) in seen):
                raise ValueError('Invalid evaluation label.')
            seen.add((item['run_id'], item['rank']))

    def get(self, workspace_id: str, batch_id: str) -> dict:
        batch = self.batches.get(workspace_id, batch_id)
        path = self.path(workspace_id, batch_id)
        try:
            with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)), 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
                    raise OSError('Evaluation labels are invalid.')
                value = json.load(stream)
            self._validate(value, workspace_id, batch_id, batch)
            return value
        except FileNotFoundError:
            return {'schema_version': 1, 'workspace_id': workspace_id, 'batch_id': batch_id,
                    'revision': None, 'labels': []}
        except (ValueError, TypeError, UnicodeError, RecursionError) as error:
            raise OSError('Evaluation labels are invalid.') from error

    def set(self, workspace_id: str, batch_id: str, run_id: str, rank: int, label: str | None,
            *, expected_revision: str | None) -> dict:
        if (not isinstance(run_id, str) or not _HEX32.fullmatch(run_id) or type(rank) is not int
                or not 1 <= rank <= 10000 or label not in ('relevant', 'irrelevant', None)
                or expected_revision is not None and (not isinstance(expected_revision, str)
                    or not _HEX32.fullmatch(expected_revision))):
            raise ValueError('Invalid evaluation label request.')
        batch = self.batches.get(workspace_id, batch_id)
        row = next((item for item in batch['rows'] if item['run_id'] == run_id), None)
        if row is None or rank > row['passage_count'] or row['status'] != 'success':
            raise ValueError('Run and passage rank do not belong to this batch.')
        path = self.path(workspace_id, batch_id)
        with FileLock(str(path) + '.lock', timeout=1):
            current = self.get(workspace_id, batch_id)
            if current['revision'] != expected_revision:
                raise WorkspaceConflict('Evaluation labels changed in another client.')
            labels = [item for item in current['labels'] if (item['run_id'], item['rank']) != (run_id, rank)]
            old = next((item for item in current['labels'] if (item['run_id'], item['rank']) == (run_id, rank)), None)
            if label is None:
                if old is None:
                    return current
            elif old is not None and old['label'] == label:
                return current
            else:
                result = self.results.read(run_id)
                if result is None:
                    raise FileNotFoundError('Saved run evidence is unavailable.')
                if result.get('workspace_id') != workspace_id or result.get('status') != 'success' or not isinstance(result.get('evidence'), list):
                    raise ValueError('Saved run evidence does not match this evaluation.')
                evidence = next((item for item in result['evidence'] if isinstance(item, dict) and item.get('rank') == rank), None)
                if (evidence is None or evidence.get('source_id') not in batch['source_ids']
                        or not isinstance(evidence.get('content'), str) or not evidence['content']
                        or type(evidence.get('start_offset')) is not int or evidence['start_offset'] < 0
                        or type(evidence.get('end_offset')) is not int or evidence['end_offset'] <= evidence['start_offset']):
                    raise ValueError('Saved passage evidence is invalid.')
                labels.append({'question_id': row['question_id'], 'run_id': run_id, 'rank': rank,
                    'source_id': evidence['source_id'], 'start_offset': evidence['start_offset'],
                    'end_offset': evidence['end_offset'],
                    'content_sha256': hashlib.sha256(evidence['content'].encode()).hexdigest(), 'label': label})
            value = {**current, 'revision': secrets.token_hex(16), 'labels': sorted(labels, key=lambda item: (item['run_id'], item['rank']))}
            self._validate(value, workspace_id, batch_id, batch)
            if len(json.dumps(value, indent=2).encode()) > _MAX_BYTES:
                raise ValueError('Evaluation labels exceed storage limit.')
            path.parent.mkdir(mode=0o700, exist_ok=True)
            write_json_atomic(path, value)
            path.chmod(0o600)
            return value
