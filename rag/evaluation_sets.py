"""Private, versioned question sets attached to shared document workspaces."""

from __future__ import annotations

import json
import os
import secrets
import stat
from pathlib import Path

from rag.workspaces import WorkspaceConflict, WorkspaceStore
from utils.persistence import write_json_atomic


class EvaluationSetStore:
    def __init__(self, workspaces: WorkspaceStore | None = None):
        self.workspaces = workspaces or WorkspaceStore()

    def _path(self, workspace_id: str) -> Path:
        return self.workspaces._workspace(workspace_id) / 'evaluation-set.json'

    def get(self, workspace_id: str) -> dict:
        self.workspaces.get(workspace_id)
        path = self._path(workspace_id)
        try:
            with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)), 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                    raise OSError('Evaluation set is invalid.')
                value = json.load(stream)
        except FileNotFoundError:
            return {'schema_version': 1, 'workspace_id': workspace_id, 'revision': None, 'questions': []}
        except (ValueError, UnicodeError) as error:
            raise OSError('Evaluation set is invalid.') from error
        if not isinstance(value, dict) or value.get('schema_version') != 1 or value.get('workspace_id') != workspace_id \
                or not isinstance(value.get('revision'), str) or len(value['revision']) > 100 \
                or not isinstance(value.get('questions'), list):
            raise OSError('Evaluation set is invalid.')
        try:
            if self._normalize(value['questions']) != value['questions']:
                raise OSError('Evaluation set is invalid.')
        except ValueError as error:
            raise OSError('Evaluation set is invalid.') from error
        return value

    @staticmethod
    def _normalize(questions: list[dict]) -> list[dict]:
        if not isinstance(questions, list) or len(questions) > 200:
            raise ValueError('Evaluation set must contain at most 200 questions.')
        seen = set()
        normalized = []
        for item in questions:
            if not isinstance(item, dict) or set(item) != {'id', 'question'}:
                raise ValueError('Each evaluation question needs an id and question.')
            identifier, question = item['id'], item['question']
            if not isinstance(identifier, str) or not 0 < len(identifier) <= 100 or any(ord(c) < 32 for c in identifier) or identifier in seen:
                raise ValueError('Evaluation question IDs must be unique and valid.')
            if not isinstance(question, str) or not 0 < len(question.strip()) <= 4000 or any(ord(c) < 32 and c not in '\n\t' for c in question):
                raise ValueError('Evaluation question text must contain 1–4000 characters.')
            normalized.append({'id': identifier, 'question': question.strip()})
            seen.add(identifier)
        return normalized

    def replace(self, workspace_id: str, questions: list[dict], *, expected_revision: str | None) -> dict:
        self.workspaces.validate_id(workspace_id)
        if expected_revision is not None and (not isinstance(expected_revision, str) or len(expected_revision) > 100):
            raise ValueError('Invalid evaluation set revision.')
        normalized = self._normalize(questions)
        with self.workspaces._lock():
            current = self.get(workspace_id)
            if current['revision'] != expected_revision:
                raise WorkspaceConflict('Evaluation set changed in another client.')
            if current['revision'] is not None and current['questions'] == normalized:
                return current
            value = {'schema_version': 1, 'workspace_id': workspace_id,
                     'revision': secrets.token_hex(16), 'questions': normalized}
            if len(json.dumps(value, indent=2).encode()) > 1024 * 1024:
                raise ValueError('Evaluation question set exceeds the 1 MiB storage limit.')
            path = self._path(workspace_id)
            write_json_atomic(path, value)
            path.chmod(0o600)
            return value
