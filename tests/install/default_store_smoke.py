"""Drive the installed CLI and MCP with their default stores and no Git."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    rag = Path(sys.executable).parent / 'rag'
    with tempfile.TemporaryDirectory(prefix='rag-default-smoke-') as directory:
        root = Path(directory)
        home = root / 'home'
        home.mkdir()
        corpus = root / 'corpus'
        corpus.mkdir()
        document = corpus / 'facts.md'
        document.write_text('The Cedar Lantern meeting is Friday.\n', encoding='utf-8')
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(('RAG_MCP_', 'XDG_', 'OPENROUTER_'))}
        env.update(HOME=str(home), PATH=str(root / 'no-git'), RAG_MCP_EMBED_PROVIDER='none',
                   RAG_MCP_RERANKER_TYPE='off', RAG_MCP_ALLOWED_ROOTS=str(corpus))
        def run(*args, input=None):
            result = subprocess.run([str(rag), *args], input=input, env=env, text=True,
                                    capture_output=True, timeout=90, check=True)
            return result.stdout
        run('sources', 'add', 'book', '--path', str(document), '--json')
        result = json.loads(run('search', 'Cedar Lantern', '--workspace', 'book', '--json'))
        assert any('Cedar Lantern' in passage['content'] for passage in result['passages'])
        requests = [
            {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'},
            {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {
                'name': 'search_workspace', 'arguments': {'workspace_id': 'book', 'query': 'Cedar Lantern',
                                                           'include_content': True}}},
        ]
        replies = [json.loads(line) for line in run('mcp', input=''.join(json.dumps(row) + '\n' for row in requests)).splitlines()]
        assert not replies[1]['result'].get('isError', False)
        assert 'Cedar Lantern' in replies[1]['result']['content'][0]['text']
        assert (home / '.local/state/ragmcp/workspaces/book').exists()


if __name__ == '__main__':
    main()
