"""One orchestration entry point for CLI, MCP, and the paired Studio gateway."""
from __future__ import annotations

import math
import json
import re
from threading import Event
import uuid
import hashlib
from filelock import FileLock, Timeout

from rag import tracing
from rag.settings import frozen_settings
from rag.answer_contract import decode_public_answer
from rag.workspaces import WorkspaceStore
from rag.run_results import RunResultStore


class RunInputError(ValueError):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def request_fingerprint(query, doc_path='', extensions=None, workspace_id='', source_ids=None, workspace_revision=None, mode='answer', answer_model=None, configuration_revision=None, conversation_id=None, conversation_revision=None):
    payload = [query, doc_path, extensions, workspace_id, source_ids, workspace_revision]
    if mode != 'answer':
        payload.append(mode)
    if answer_model is not None or configuration_revision is not None:
        payload.append({'answer_model': answer_model, 'configuration_revision': configuration_revision})
    if conversation_id is not None:
        payload.append({'conversation_id': conversation_id, 'conversation_revision': conversation_revision})
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


# Text extracted from these formats has its own line breaks, so its lines are not
# lines of the file a reader would open.
_EXTRACTED_TYPES = {'pdf', 'docx', 'html'}


def public_response(result: dict) -> str:
    return json.dumps({'result': result['answer']} if result['status'] == 'success'
                      else {'error': result['error']})


def retrieval_response(result: dict, *, include_content: bool = False) -> dict:
    """Project a private retrieval run for a local client or MCP host."""
    if result['status'] != 'success':
        return {'error': result['error'], 'run_id': result['run_id'], 'code': result.get('code')}
    passages = []
    for item in result.get('retrieved_passages', result['evidence']):
        passage = {key: item[key] for key in
                   ('rank', 'source_id', 'name', 'source_type', 'start_offset', 'end_offset')}
        passage['location_verified'] = item.get('location_verified', type(item['start_offset']) is int)
        passage.update({key: item[key] for key in ('start_line', 'end_line') if key in item})
        if include_content:
            passage.update(content=item['content'], score=item['score'])
        passages.append(passage)
    return {'schema_version': 1, 'run_id': result['run_id'],
            'workspace_id': result['workspace_id'],
            'workspace_revision': result['workspace_revision'],
            'configuration_revision': result['configuration_revision'],
            'passage_count': len(passages), 'passages': passages}


def execute_run(query: str, doc_path: str = "", extensions: list[str] | None = None,
                workspace_id: str = "", *, client: str = "cli", cancelled: Event | None = None,
                run_id: str | None = None, workspace_store: WorkspaceStore | None = None,
                source_ids: list[str] | None = None, workspace_revision: str | None = None,
                result_store: RunResultStore | None = None, mode: str = 'answer',
                answer_model: str | None = None, configuration_revision: str | None = None,
                request_hash_override: str | None = None) -> dict:
    """Public adapters select answer/error; private callers can inspect actual evidence."""
    from rag import server
    run_id = run_id or uuid.uuid4().hex
    if request_hash_override is not None and (not isinstance(request_hash_override, str)
            or not re.fullmatch(r'[0-9a-f]{64}', request_hash_override)):
        raise ValueError('Invalid request fingerprint.')
    request_hash = request_hash_override if request_hash_override is not None else request_fingerprint(
        query, doc_path, extensions, workspace_id, source_ids, workspace_revision, mode,
        answer_model, configuration_revision)
    evidence = []
    retrieved_passages = []
    snapshot = None
    source_paths = {}
    selected_files = None

    def observe(name, passages):
        if name != 'synthesis_evidence' or snapshot is None:
            return
        evidence.clear()
        retrieved_passages.clear()
        for rank, passage in enumerate(passages, 1):
            path = passage.get('file_path', passage.get('source'))
            source = source_paths.get(str(path))
            text = passage.get('original_content') or passage.get('content')
            if source is None or not isinstance(text, str):
                continue
            score = passage.get('score')
            row = {'rank': rank, 'source_id': source['id'], 'name': source['name'],
                   'source_type': source['sourceType'], 'content': text,
                   'start_offset': None, 'end_offset': None, 'location_verified': False,
                   'score': score if type(score) in (int, float) and math.isfinite(score) else None}
            content = source['content']
            start, end = passage.get('start_offset'), passage.get('end_offset')
            exact = (type(start) is int and type(end) is int and
                     0 <= start < end <= len(content) and
                     content[start:end].replace('\f', ' ') == text)
            if not exact:
                # Older indexes have only line locations. Never guess when a
                # repeated passage could point at another occurrence.
                normalized = content.replace('\f', ' ')
                line = passage.get('start_line', 0)
                lower = sum(len(part) for part in normalized.splitlines(keepends=True)[:line - 1]) if type(line) is int and line > 0 else 0
                start = normalized.find(text, lower)
                if start < 0:
                    retrieved_passages.append(row)
                    continue
                next_match = normalized.find(text, start + 1)
                if next_match >= 0 and (not line or normalized.count('\n', start, next_match) == 0):
                    retrieved_passages.append(row)
                    continue
                end = start + len(text)
            original = content[start:end]
            # JavaScript source navigation indexes UTF-16 code units.
            browser_start = len(content[:start].encode('utf-16-le')) // 2
            row.update(content=original, start_offset=browser_start,
                       end_offset=browser_start + len(original.encode('utf-16-le')) // 2,
                       location_verified=True)
            if source['sourceType'] not in _EXTRACTED_TYPES:
                # 1-based, inclusive: the lines a coding agent or a person opens the file at.
                row.update(start_line=content.count('\n', 0, start) + 1,
                           end_line=content.count('\n', 0, max(start, end - 1)) + 1)
            retrieved_passages.append(row)
            evidence.append({key: value for key, value in row.items()
                             if key not in ('location_verified', 'start_line', 'end_line')})

    result = {'schema_version': 1, 'run_id': run_id, 'workspace_id': workspace_id or None,
              'workspace_revision': None, 'client': client, 'status': 'error',
              'mode': mode,
              'request_hash': request_hash,
              'answer': None, 'error': None, 'evidence': evidence,
              'retrieved_passages': retrieved_passages}
    try:
        store = workspace_store or WorkspaceStore()
        store.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tracing.execution_context(observe, cancelled), tracing.request_trace(run_id), FileLock(store.root / '.execution.lock', timeout=0), frozen_settings():
            tracing.annotate(client=client, workspace_id=workspace_id or None)
            from rag.provenance import runtime_manifest
            try:
                result['runtime_manifest'] = runtime_manifest()
            except Exception:
                result['runtime_manifest_status'] = 'unavailable'
                tracing.annotate(runtime_manifest_status='unavailable')
            else:
                tracing.annotate(runtime_manifest_sha256=result['runtime_manifest']['sha256'])
            if mode not in {'answer', 'retrieve'}:
                raise RunInputError('Unsupported run mode.', 'invalid_mode')
            tracing.annotate(mode=mode)
            from rag.control import read_config
            result['configuration'] = read_config()
            result['configuration_revision'] = hashlib.sha256(json.dumps(result['configuration'], sort_keys=True).encode()).hexdigest()
            tracing.annotate(configuration_revision=result['configuration_revision'])
            if configuration_revision is not None and configuration_revision != result['configuration_revision']:
                raise RunInputError('Engine settings changed during comparison. Run it again with the current settings.', 'configuration_conflict')
            if answer_model is not None:
                if not isinstance(answer_model, str) or not answer_model.strip() or len(answer_model) > 200 or any(ord(c) < 32 for c in answer_model):
                    raise RunInputError('Invalid answer model.', 'invalid_model')
                result['configuration']['retrieval']['answer_model'] = answer_model
            tracing.annotate(embedding_model=result['configuration']['embedding']['model'],
                             model=result['configuration']['retrieval']['answer_model'],
                             reranker_model=result['configuration']['retrieval']['reranker_model'])
            if workspace_id and not doc_path:
                try:
                    # Snapshot and path must refer to the same immutable generation.
                    snapshot = store.snapshot(workspace_id)
                    if workspace_revision is not None and snapshot['revision'] != workspace_revision:
                        raise RunInputError('Workspace changed. Sync sources before querying.', 'workspace_conflict')
                    tracing.annotate(workspace_revision=snapshot['revision'])
                    root = store.root / workspace_id / 'generations' / snapshot['revision'] / 'documents'
                    source_paths = {str(root / store.document_filename(item['id'], item['sourceType'])): item for item in snapshot['documents']}
                    if source_ids is not None:
                        available = {item['id'] for item in snapshot['documents']}
                        if not source_ids or not set(source_ids).issubset(available):
                            raise RunInputError('Select available workspace sources before querying.', 'invalid_sources')
                        selected_files = [path for path, source in source_paths.items() if source['id'] in source_ids]
                    else:
                        # An omitted selection means every saved source, including
                        # the scope Studio sends explicitly for its default view.
                        selected_files = list(source_paths)
                    doc_path = str(root)
                    result['workspace_revision'] = snapshot['revision']
                    workspace_id = ''
                except RunInputError:
                    raise
                except (ValueError, OSError):
                    pass  # Existing scope validation produces the public typed failure.
            response = server._query_knowledge_base(query, doc_path, extensions, workspace_id,
                                                    selected_files=selected_files, cite_sources=client == 'studio', retrieval_only=mode == 'retrieve', answer_model=answer_model)
            with tracing.span('public.boundary'):
                answer, failed = decode_public_answer(response)
                tracing.annotate(is_error=failed)
            if failed:
                tracing.fail_current()
                result['error'] = answer
            else:
                result.update(status='success', answer=answer if mode == 'answer' else None)
            tracing.content('public_result', response)
    except tracing.RunCancelled:
        result.update(status='cancelled', error='Query cancelled.', answer=None)
        evidence.clear()
        retrieved_passages.clear()
    except RunInputError as error:
        result.update(error=str(error), code=error.code)
    except Timeout:
        result.update(error='Another engine query is running. Wait for it to finish or cancel it.', code='engine_busy')
    except Exception:
        result.update(status='error', error='Query execution failed. Check the local engine diagnostics.', answer=None)
        evidence.clear()
        retrieved_passages.clear()
    try:
        (result_store or RunResultStore()).save(result)
    except (OSError, ValueError):
        result['persistence_error'] = 'Run result could not be saved; the answer is available in this response.'
    return result
