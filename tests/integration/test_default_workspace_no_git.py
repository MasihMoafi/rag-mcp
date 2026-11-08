import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(('allowed', 'configured'), [(False, ''), (True, ''), (True, 'hidden'), (True, 'symlink')])
def test_default_workspace_search_without_git(tmp_path, allowed, configured):
    home = tmp_path / 'home'
    home.mkdir()
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('RAG_MCP_', 'XDG_', 'OPENROUTER_'))}
    env.update(HOME=str(home), PATH=str(tmp_path / 'no-bin'),
               RAG_MCP_EMBED_PROVIDER='none', RAG_MCP_RERANKER_TYPE='off', RAG_MCP_BACKEND='qdrant')
    if allowed:
        env['RAG_MCP_ALLOWED_ROOTS'] = str(corpus)
    if configured:
        if configured == 'symlink':
            target = tmp_path / 'actual-store'
            target.mkdir()
            store_path = tmp_path / 'linked-store'
            store_path.symlink_to(target, target_is_directory=True)
        else:
            store_path = home / '.local' / '__pycache__' / 'workspaces'
        env['RAG_MCP_WORKSPACE_DIR'] = str(store_path)
    script = '''
import json
from rag import core
from rag.workspaces import WorkspaceStore
from rag.execution import execute_run
class TestEmbedding:
    model_name = 'fixture'
    def embed_query(self, text): return [1.0, 0.0]
    def embed_documents(self, texts): return [[1.0, 0.0] for _ in texts]
core.build_embedder = lambda *args, **kwargs: TestEmbedding()
store = WorkspaceStore()
store.replace('book', 'Book', [{'id': 'a', 'name': 'a.txt', 'source_type': 'text', 'content': 'Cedar Lantern meets Friday.'}])
result = execute_run('Cedar Lantern', workspace_id='book', mode='retrieve')
print(json.dumps(result))
'''
    try:
        run = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        pytest.fail((error.stderr or b'').decode())
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.splitlines()[-1])
    assert result['status'] == 'success', result['error'] + '\n' + run.stderr
    assert any('Cedar Lantern' in item['content'] for item in result['retrieved_passages'])
    store = store_path if configured else home / '.local/state/ragmcp/workspaces'
    assert (store / 'book').exists()
