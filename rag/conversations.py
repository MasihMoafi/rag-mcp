"""Private, revisioned conversations anchored to saved workspace runs."""
from __future__ import annotations

import json
import math
import os
import re
import secrets
import stat
import time
import uuid

from rag.run_results import RunResultStore
from rag.workspaces import WorkspaceConflict, WorkspaceStore
from utils.persistence import write_json_atomic

_ID = re.compile(r'[0-9a-f]{32}')
_REVISION = re.compile(r'[0-9a-f]{64}')
_MAX_BYTES = 8 * 1024 * 1024
_MAX_TURNS = 100
_MAX_CONVERSATIONS = 1000
_UNSET = object()


class ConversationStore:
    def __init__(self, workspaces: WorkspaceStore | None = None, results: RunResultStore | None = None):
        self.workspaces = workspaces or WorkspaceStore()
        self.results = results or RunResultStore()

    def path(self, workspace_id: str, conversation_id: str):
        self.workspaces.validate_id(workspace_id)
        if not isinstance(conversation_id, str) or not _ID.fullmatch(conversation_id):
            raise ValueError('Invalid conversation ID.')
        directory = self.workspaces._workspace(workspace_id) / 'conversations'
        if directory.is_symlink():
            raise OSError('Conversation storage is invalid.')
        return directory / f'{conversation_id}.json'

    @staticmethod
    def _validate(value, workspace_id, conversation_id):
        if (not isinstance(value, dict) or set(value) != {'schema_version', 'workspace_id', 'conversation_id',
                'revision', 'created_unix', 'updated_unix', 'turns'}
                or value['schema_version'] != 1 or value['workspace_id'] != workspace_id
                or value['conversation_id'] != conversation_id or not isinstance(value['revision'], str)
                or not _ID.fullmatch(value['revision']) or not isinstance(value['turns'], list)
                or not 1 <= len(value['turns']) <= _MAX_TURNS):
            raise ValueError('Invalid conversation.')
        for field in ('created_unix', 'updated_unix'):
            number = value[field]
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError('Invalid conversation time.')
        if value['updated_unix'] < value['created_unix']:
            raise ValueError('Invalid conversation time.')
        seen = set()
        for turn in value['turns']:
            if (not isinstance(turn, dict) or set(turn) != {'run_id', 'question', 'answer', 'workspace_revision',
                    'configuration_revision', 'source_ids', 'evidence'}
                    or not isinstance(turn['run_id'], str) or not _ID.fullmatch(turn['run_id'])
                    or turn['run_id'] in seen or not isinstance(turn['question'], str)
                    or not 1 <= len(turn['question'].strip()) <= 4000
                    or not isinstance(turn['answer'], str) or not 1 <= len(turn['answer'].strip()) <= 100000
                    or (turn['workspace_revision'] is not None and not isinstance(turn['workspace_revision'], str))
                    or (turn['configuration_revision'] is not None and (not isinstance(turn['configuration_revision'], str)
                        or not _REVISION.fullmatch(turn['configuration_revision'])))
                    or not isinstance(turn['source_ids'], list) or len(turn['source_ids']) > 500
                    or not all(isinstance(source, str) and 0 < len(source) <= 256 for source in turn['source_ids'])
                    or len(set(turn['source_ids'])) != len(turn['source_ids'])
                    or not isinstance(turn['evidence'], list) or len(turn['evidence']) > 100):
                raise ValueError('Invalid conversation turn.')
            seen.add(turn['run_id'])
            ranks = set()
            for item in turn['evidence']:
                if (not isinstance(item, dict) or set(item) != {'rank', 'source_id', 'name', 'content',
                        'start_offset', 'end_offset', 'score'}
                        or type(item['rank']) is not int or not 1 <= item['rank'] <= 10000 or item['rank'] in ranks
                        or item['source_id'] not in turn['source_ids']
                        or not isinstance(item['name'], str) or len(item['name']) > 1000
                        or not isinstance(item['content'], str) or not 0 < len(item['content']) <= 100000
                        or type(item['start_offset']) is not int or item['start_offset'] < 0
                        or type(item['end_offset']) is not int or item['end_offset'] <= item['start_offset']
                        or (item['score'] is not None and (type(item['score']) not in (int, float)
                            or not math.isfinite(item['score'])))):
                    raise ValueError('Invalid conversation evidence.')
                ranks.add(item['rank'])

    def get(self, workspace_id: str, conversation_id: str) -> dict:
        self.workspaces.get(workspace_id)
        path = self.path(workspace_id, conversation_id)
        try:
            with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)), 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
                    raise OSError('Conversation is invalid.')
                value = json.load(stream)
            self._validate(value, workspace_id, conversation_id)
            return value
        except (ValueError, TypeError, UnicodeError, RecursionError) as error:
            raise OSError('Conversation is invalid.') from error

    def list(self, workspace_id: str, *, limit=20, offset=0) -> dict:
        self.workspaces.get(workspace_id)
        if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or not 0 <= offset <= 10000:
            raise ValueError('Invalid conversation list bounds.')
        directory = self.path(workspace_id, '0' * 32).parent
        paths = sorted((path for path in directory.glob('*.json') if _ID.fullmatch(path.stem)),
                       key=lambda path: path.lstat().st_mtime_ns, reverse=True)
        conversations, skipped = [], 0
        for path in paths[offset:offset + limit]:
            try:
                value = self.get(workspace_id, path.stem)
                conversations.append({'conversation_id': value['conversation_id'], 'revision': value['revision'],
                                      'created_unix': value['created_unix'], 'updated_unix': value['updated_unix'],
                                      'turn_count': len(value['turns']), 'last_question': value['turns'][-1]['question']})
            except OSError:
                skipped += 1
        return {'schema_version': 1, 'workspace_id': workspace_id, 'conversations': conversations,
                'skipped': skipped, 'next_offset': offset + limit if len(paths) > offset + limit else None}

    def append(self, workspace_id: str, conversation_id: str, run_id: str, question: str,
               *, expected_revision: str | None) -> dict:
        self.workspaces.validate_id(workspace_id)
        path = self.path(workspace_id, conversation_id)
        if (not isinstance(run_id, str) or not _ID.fullmatch(run_id)
                or not isinstance(question, str) or not 1 <= len(question.strip()) <= 4000
                or expected_revision is not None and (not isinstance(expected_revision, str)
                    or not _ID.fullmatch(expected_revision))):
            raise ValueError('Invalid conversation turn request.')
        question = question.strip()
        with self.workspaces._lock():
            self.workspaces.get(workspace_id)
            try:
                current = self.get(workspace_id, conversation_id)
            except FileNotFoundError:
                current = None
            if current is not None:
                previous = next((turn for turn in current['turns'] if turn['run_id'] == run_id), None)
                if previous is not None:
                    if previous['question'] != question:
                        raise ValueError('Run already belongs to a different conversation turn.')
                    return current
            if (current['revision'] if current is not None else None) != expected_revision:
                raise WorkspaceConflict('Conversation changed in another client.')
            if current is not None and len(current['turns']) >= _MAX_TURNS:
                raise ValueError('Conversation has reached its turn limit. Start a new chat.')
            if current is None and sum(1 for item in path.parent.glob('*.json') if _ID.fullmatch(item.stem)) >= _MAX_CONVERSATIONS:
                raise ValueError('Workspace has reached its conversation limit.')
            result = self.results.read(run_id)
            if result is None:
                raise FileNotFoundError('Saved run result is unavailable.')
            if (result.get('workspace_id') != workspace_id or result.get('status') != 'success'
                    or not isinstance(result.get('answer'), str) or not isinstance(result.get('evidence'), list)):
                raise ValueError('Saved run is not a successful answer in this workspace.')
            if len(result['evidence']) > 100 or not all(isinstance(item, dict) for item in result['evidence']):
                raise ValueError('Saved run evidence is invalid.')
            evidence = [{field: item.get(field) for field in ('rank', 'source_id', 'name', 'content',
                         'start_offset', 'end_offset', 'score')} for item in result['evidence']]
            sources = list(dict.fromkeys(item['source_id'] for item in evidence if isinstance(item['source_id'], str)))
            turn = {'run_id': run_id, 'question': question, 'answer': result['answer'],
                    'workspace_revision': result.get('workspace_revision'),
                    'configuration_revision': result.get('configuration_revision'),
                    'source_ids': sources, 'evidence': evidence}
            now = time.time()
            value = {'schema_version': 1, 'workspace_id': workspace_id, 'conversation_id': conversation_id,
                     'revision': secrets.token_hex(16), 'created_unix': current['created_unix'] if current else now,
                     'updated_unix': now, 'turns': [*(current['turns'] if current else []), turn]}
            self._validate(value, workspace_id, conversation_id)
            if len(json.dumps(value, ensure_ascii=True, indent=2).encode()) > _MAX_BYTES:
                raise ValueError('Conversation exceeds storage limit. Start a new chat.')
            path.parent.mkdir(mode=0o700, exist_ok=True)
            path.parent.chmod(0o700)
            write_json_atomic(path, value)
            path.chmod(0o600)
            return value


def prepare_conversation_query(question: str, workspace_id: str, conversation_id: str, *,
                               expected_revision=_UNSET, store: ConversationStore) -> tuple[str, str | None]:
    """Check a conversation revision and form the shared retrieval question."""
    store.path(workspace_id, conversation_id)
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 4000:
        raise ValueError('Conversation question must contain 1–4000 characters.')
    question = question.strip()
    try:
        current = store.get(workspace_id, conversation_id)
    except FileNotFoundError:
        store.workspaces.get(workspace_id)
        current = None
    revision = current['revision'] if current else None
    if expected_revision is not _UNSET and expected_revision != revision:
        raise WorkspaceConflict('Conversation changed. Reload before querying.')
    query = (
        f"Previous question (context only, do not answer): {current['turns'][-1]['question']}\n"
        f"Current question (answer this question only): {question}"
        if current else question
    )
    return query, revision


def run_conversation_query(question: str, workspace_id: str, conversation_id: str, *, client='cli',
                           expected_revision=_UNSET, store: ConversationStore | None = None,
                           result_store: RunResultStore | None = None, executor=None) -> dict:
    """Answer using the recorded previous question, then attach the saved run."""
    from rag.execution import execute_run

    store = store or ConversationStore(results=result_store)
    query, revision = prepare_conversation_query(question, workspace_id, conversation_id,
                                                  expected_revision=expected_revision, store=store)
    question = question.strip()
    run_id = uuid.uuid4().hex
    result = (executor or execute_run)(query, workspace_id=workspace_id, client=client,
                                       run_id=run_id, result_store=result_store or store.results,
                                       workspace_store=store.workspaces)
    if (not isinstance(result, dict) or result.get('run_id') != run_id
            or result.get('status') == 'success' and (not isinstance(result.get('answer'), str)
                or not result['answer'].strip())):
        raise OSError('Engine returned an invalid conversation run.')
    if result.get('status') != 'success':
        return {'error': result.get('error') or 'Conversation query failed.',
                'run_id': run_id, 'conversation_id': conversation_id}
    value = {'result': result.get('answer'), 'run_id': run_id, 'conversation_id': conversation_id}
    try:
        saved = store.append(workspace_id, conversation_id, run_id, question, expected_revision=revision)
        value['conversation_revision'] = saved['revision']
    except WorkspaceConflict:
        value['conversation_error'] = 'Conversation changed while the answer ran. Reload it and append this saved run.'
    except (FileNotFoundError, ValueError, OSError):
        value['conversation_error'] = 'Answer ran but could not be saved to the conversation. Inspect the saved run and retry.'
    return value
