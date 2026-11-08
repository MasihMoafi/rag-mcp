"""Publication must wait for quality and tests of the artifacts being published."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def publication_has_gates(jobs, publication):
    seen = set()

    def visit(job):
        if job in seen:
            return
        seen.add(job)
        needs = jobs[job].get('needs', [])
        for dependency in [needs] if isinstance(needs, str) else needs:
            visit(dependency)

    visit(publication)
    return {'build', 'quality', 'installed-artifact'} <= seen


def test_both_publications_depend_on_quality_and_installed_artifacts():
    release = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())
    quality = yaml.safe_load((ROOT / '.github/workflows/python-quality.yml').read_text())
    jobs = release['jobs']
    assert {'keyless', 'lint', 'types', 'default-store-no-git'} <= quality['jobs'].keys()
    assert set(quality['jobs']['keyless']['strategy']['matrix']['python']) == {'3.10', '3.13'}
    assert jobs['quality']['uses'] == './.github/workflows/python-quality.yml'
    for name in ('publish-pypi', 'publish-npm'):
        assert publication_has_gates(jobs, name)
    steps = jobs['installed-artifact']['steps']
    commands = '\n'.join(step.get('run', '') for step in steps)
    assert 'sha256sum -c SHA256SUMS' in commands
    assert 'tests/install/smoke.sh' in commands
    assert any(step.get('with', {}).get('name') == 'release-artifacts' for step in steps)
    weakened = dict(jobs)
    weakened['publish-pypi'] = {'needs': ['build']}
    assert not publication_has_gates(weakened, 'publish-pypi')
    assert not publication_has_gates(weakened, 'publish-npm')


def test_manual_release_rehearsal_runs_checks_but_never_publishes():
    release = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())
    triggers = release.get('on', release.get(True))
    assert 'workflow_dispatch' in triggers
    jobs = release['jobs']
    for name in ('quality', 'build', 'installed-artifact'):
        assert "github.event_name == 'workflow_dispatch'" in jobs[name]['if']
    for name in ('publish-pypi', 'publish-npm'):
        condition = jobs[name]['if']
        assert "github.event_name == 'push'" in condition
        assert "vars.PUBLISH_ENABLED == 'true'" in condition
    build_commands = '\n'.join(step.get('run', '') for step in jobs['build']['steps'])
    assert 'check_release_artifacts.py' in build_commands
    assert 'workflow_dispatch' in build_commands


def test_semantic_search_survives_a_lexical_miss(tmp_path, monkeypatch):
    import json
    import re
    import httpx
    import openai
    from rag import fetch, settings
    from rag.execution import execute_run
    from rag.workspaces import WorkspaceStore

    text = (ROOT / 'tests/install/facts.md').read_text()
    query = 'nocturnal beacon'
    assert set(re.findall(r'\w+', query)).isdisjoint(re.findall(r'\w+', text.lower()))
    monkeypatch.setenv('HOME', str(tmp_path))
    for name in ('RAG_MCP_CONFIG', 'RAG_MCP_DATA_DIR', 'RAG_MCP_WORKSPACE_DIR', 'RAG_MCP_TRACE_DIR', 'XDG_STATE_HOME', 'XDG_DATA_HOME', 'XDG_CONFIG_HOME'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(fetch.RAG_CONFIG_V2, 'backend', 'qdrant')
    monkeypatch.setenv('RAG_MCP_RERANKER_TYPE', 'off')
    settings.update_settings({'embed_provider': 'openai_compatible', 'embed_model': 'mock-embed',
                              'embed_base_url': 'https://semantic.test/v1', 'embed_api_key': 'fixture'})
    requests = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        inputs = body['input'] if isinstance(body['input'], list) else [body['input']]
        return httpx.Response(200, json={'model': 'mock-embed', 'data': [
            {'index': index, 'embedding': [1.0, 0.0]} for index, _ in enumerate(inputs)]})
    real_client = openai.OpenAI
    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: real_client(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    store = WorkspaceStore()
    store.replace('smoke', 'Smoke', [{'id': 'facts', 'name': 'facts.md', 'source_type': 'text', 'content': text}])
    try:
        result = execute_run(query, workspace_id='smoke', mode='retrieve')
        assert result['status'] == 'success', result['error']
        assert any('Cedar Lantern' in passage['content'] for passage in result['retrieved_passages'])
        assert any(query in (body['input'] if isinstance(body['input'], list) else [body['input']]) for body in requests)
    finally:
        fetch.reset_pipelines()
