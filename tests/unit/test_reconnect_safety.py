"""Reconnecting an agent to rag never loses its existing `rag` registration.

Each fake client keeps a registry. Adding a name that already exists fails, as Claude Code and Codex do; a client set
to reject the new command fails the add for another reason. `get` reports the old entry in the client's own format.
"""

import io
import json
import subprocess

import pytest

from rag import cli_agents

AGENTS = ["codex", "claude", "copilot", "gemini", "agy"]
READABLE = {
    "codex",
    "claude",
}  # their current entry can be read back, so it can be restored


def fake_client(monkeypatch, agent, *, reject_new):
    registry = {"rag": ["old-server", "mcp"]}
    calls = []
    monkeypatch.setattr(cli_agents.shutil, "which", lambda *a, **k: agent)
    monkeypatch.setattr(cli_agents, "server_command", lambda _: ["new-server", "mcp"])
    if agent == 'claude':
        monkeypatch.setattr(cli_agents, '_prepare_claude_recovery', lambda *_: (lambda: True, lambda: None))
        monkeypatch.setattr(cli_agents, '_claude_entry', lambda *_: {
            'type': 'stdio', 'command': 'old-server', 'args': ['mcp']})

    def run(args, **kwargs):
        calls.append(list(args))
        if "get" in args:
            if "rag" not in registry:
                return subprocess.CompletedProcess(args, 1, "", "not found")
            command, *rest = registry["rag"]
            if agent == "codex":
                out = json.dumps({"transport": {"command": command, "args": rest}})
            else:
                out = f"rag:\n  Type: stdio\n  Command: {command}\n  Args: {' '.join(rest)}\n"
            return subprocess.CompletedProcess(args, 0, out, "")
        if "remove" in args:
            registry.pop("rag", None)
            return subprocess.CompletedProcess(args, 0, "", "")
        if "add-json" in args:
            entry = json.loads(args[args.index("rag") + 1])
            if 'rag' in registry or (entry['command'] == 'new-server' and reject_new):
                return subprocess.CompletedProcess(args, 1, '', 'add failed')
            registry["rag"] = [entry["command"], *entry["args"]]
            return subprocess.CompletedProcess(args, 0, "", "")
        command = "new-server" if "new-server" in args else "old-server"
        if "rag" in registry and agent != 'codex':
            return subprocess.CompletedProcess(args, 1, "", "rag already exists")
        if command == "new-server" and reject_new:
            return subprocess.CompletedProcess(args, 1, "", "add failed")
        registry["rag"] = [command, "mcp"]
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(cli_agents.subprocess, "run", run)
    return registry, calls


@pytest.mark.parametrize("agent", AGENTS)
def test_failed_reconnect_keeps_the_old_registration(monkeypatch, agent):
    registry, calls = fake_client(monkeypatch, agent, reject_new=True)
    assert cli_agents.connect_agent(agent, {}, io.StringIO(), io.StringIO()) != 0
    assert registry["rag"] == ["old-server", "mcp"]
    if agent == 'codex':
        assert not any('remove' in call for call in calls)


@pytest.mark.parametrize("agent", sorted(READABLE))
def test_reconnect_replaces_an_outdated_registration(monkeypatch, agent):
    registry, _ = fake_client(monkeypatch, agent, reject_new=False)
    assert cli_agents.connect_agent(agent, {}, io.StringIO(), io.StringIO()) == 0
    assert registry["rag"] == ["new-server", "mcp"]


@pytest.mark.parametrize("agent", sorted(set(AGENTS) - READABLE))
def test_unreadable_clients_never_remove_the_old_registration(monkeypatch, agent):
    registry, calls = fake_client(monkeypatch, agent, reject_new=False)
    stderr = io.StringIO()
    assert cli_agents.connect_agent(agent, {}, io.StringIO(), stderr) != 0
    assert registry["rag"] == ["old-server", "mcp"]
    assert not any("remove" in call for call in calls)
    assert "already exists" in stderr.getvalue()


def test_claude_failed_replacement_restores_complete_json_entry(tmp_path, monkeypatch):
    entry = {'type': 'stdio', 'command': 'old-server', 'args': ['mcp'],
             'env': {'RAG_MCP_ALLOWED_ROOTS': '/restricted'}, 'cwd': '/work'}
    (tmp_path / '.claude.json').write_text(json.dumps({'mcpServers': {'rag': entry}}))
    registry = {'rag': entry}
    monkeypatch.setattr(cli_agents.shutil, 'which', lambda *a, **k: 'claude')
    monkeypatch.setattr(cli_agents, 'server_command', lambda _: ['new-server', 'mcp'])

    def run(args, **kwargs):
        if 'get' in args:
            return subprocess.CompletedProcess(args, 0, 'Type: stdio\nCommand: old-server\nArgs: mcp\n', '')
        if 'remove' in args:
            registry.pop('rag', None)
            return subprocess.CompletedProcess(args, 0, '', '')
        if 'add-json' in args:
            value = json.loads(args[args.index('rag') + 1])
            if value['command'] == 'new-server':
                return subprocess.CompletedProcess(args, 1, '', 'replacement failed')
            registry['rag'] = value
            return subprocess.CompletedProcess(args, 0, '', '')
        return subprocess.CompletedProcess(args, 1, '', 'replacement failed')

    monkeypatch.setattr(cli_agents.subprocess, 'run', run)
    assert cli_agents.connect_claude({'HOME': str(tmp_path)}, io.StringIO(), io.StringIO()) != 0
    assert registry['rag'] == entry


def test_failed_restore_reports_exact_entry(tmp_path, monkeypatch):
    entry = {'type': 'stdio', 'command': 'old-server', 'args': [],
             'env': {'RAG_MCP_ALLOWED_ROOTS': '/restricted'}}
    (tmp_path / '.claude.json').write_text(json.dumps({'mcpServers': {'rag': entry}}))
    monkeypatch.setattr(cli_agents.shutil, 'which', lambda *a, **k: 'claude')
    monkeypatch.setattr(cli_agents, 'server_command', lambda _: ['new-server'])
    config_path = tmp_path / '.claude.json'
    def run(args, **kwargs):
        if 'remove' in args:
            config_path.write_text(json.dumps({'mcpServers': {}, 'other_setting': 'concurrent change'}))
            return subprocess.CompletedProcess(args, 0, '', '')
        if 'add-json' in args and json.loads(args[args.index('rag') + 1])['command'] == 'old-server':
            raise FileNotFoundError('client disappeared')
        return subprocess.CompletedProcess(args, 1, '', 'failed')
    monkeypatch.setattr(cli_agents.subprocess, 'run', run)
    stderr = io.StringIO()
    assert cli_agents.connect_claude({'HOME': str(tmp_path)}, io.StringIO(), stderr) != 0
    assert json.loads(config_path.read_text())['mcpServers']['rag'] == entry
    assert json.loads(config_path.read_text())['other_setting'] == 'concurrent change'
    assert not list(tmp_path.glob('..claude.json.rag-backup-*'))
    assert 'restored from its backup' in stderr.getvalue()


def test_failed_file_recovery_gives_backup_location_without_secrets(tmp_path, monkeypatch):
    entry = {'type': 'stdio', 'command': 'old-server', 'args': ['mcp']}
    config_path = tmp_path / '.claude.json'
    config_path.write_text(json.dumps({'secret': 'private-token', 'mcpServers': {'rag': entry}}))
    monkeypatch.setattr(cli_agents.shutil, 'which', lambda *a, **k: 'claude')
    monkeypatch.setattr(cli_agents, 'server_command', lambda _: ['new-server', 'mcp'])
    def run(args, **kwargs):
        if 'remove' in args:
            config_path.write_text('{broken')
            return subprocess.CompletedProcess(args, 0, '', '')
        return subprocess.CompletedProcess(args, 1, '', 'failed')
    monkeypatch.setattr(cli_agents.subprocess, 'run', run)
    stderr = io.StringIO()
    assert cli_agents.connect_claude({'HOME': str(tmp_path)}, io.StringIO(), stderr) != 0
    assert not list(tmp_path.glob('..claude.json.rag-backup-*'))
    assert 'Restore mcpServers.rag manually' in stderr.getvalue()
    assert 'private-token' not in stderr.getvalue()


@pytest.mark.parametrize('cwd', [None, '/restricted/work'])
def test_codex_keeps_environment_and_refuses_unrestorable_cwd(monkeypatch, cwd):
    entry = {'name': 'rag', 'enabled': True, 'transport': {
        'type': 'stdio', 'command': 'old-server', 'args': ['mcp'],
        'env': {'RAG_MCP_ALLOWED_ROOTS': '/restricted'}, 'cwd': cwd, 'env_vars': []}}
    registry = {'rag': entry}
    monkeypatch.setattr(cli_agents.shutil, 'which', lambda *a, **k: 'codex')
    monkeypatch.setattr(cli_agents, 'server_command', lambda _: ['new-server', 'mcp'])

    def run(args, **kwargs):
        if 'get' in args:
            return subprocess.CompletedProcess(args, 0, json.dumps(entry), '')
        if 'remove' in args:
            registry.pop('rag', None)
            return subprocess.CompletedProcess(args, 0, '', '')
        if 'old-server' in args:
            assert 'RAG_MCP_ALLOWED_ROOTS=/restricted' in args
            registry['rag'] = entry
            return subprocess.CompletedProcess(args, 0, '', '')
        return subprocess.CompletedProcess(args, 1, '', 'replacement failed')

    monkeypatch.setattr(cli_agents.subprocess, 'run', run)
    assert cli_agents.connect_codex({}, io.StringIO(), io.StringIO()) != 0
    assert registry['rag'] == entry


def test_codex_failed_restore_reports_exact_entry(monkeypatch):
    entry = {'name': 'rag', 'transport': {'type': 'stdio', 'command': 'old-server',
             'args': ['mcp'], 'env': {'RAG_MCP_ALLOWED_ROOTS': '/restricted'}}}
    monkeypatch.setattr(cli_agents.shutil, 'which', lambda *a, **k: 'codex')
    monkeypatch.setattr(cli_agents, 'server_command', lambda _: ['new-server', 'mcp'])

    def run(args, **kwargs):
        if 'get' in args:
            return subprocess.CompletedProcess(args, 0, json.dumps(entry), '')
        return subprocess.CompletedProcess(args, 0 if 'remove' in args else 1, '', 'failed')

    monkeypatch.setattr(cli_agents.subprocess, 'run', run)
    stderr = io.StringIO()
    assert cli_agents.connect_codex({}, io.StringIO(), stderr) != 0
    assert 'left unchanged' in stderr.getvalue()


@pytest.mark.parametrize('agent', ['codex', 'claude'])
def test_replacement_preserves_restriction_and_refuses_secrets(agent, monkeypatch):
    entry = {'command': 'old-server', 'args': ['mcp'], 'env': {'RAG_MCP_ALLOWED_ROOTS': '/restricted'}}
    calls = []
    monkeypatch.setattr(cli_agents.shutil, 'which', lambda *a, **k: agent)
    monkeypatch.setattr(cli_agents, 'server_command', lambda _: ['new-server', 'mcp'])
    monkeypatch.setattr(cli_agents, '_claude_entry', lambda *_: entry)
    monkeypatch.setattr(cli_agents, '_prepare_claude_recovery', lambda *_: (lambda: True, lambda: None))
    def run(args, **kwargs):
        calls.append(args)
        if 'get' in args:
            return subprocess.CompletedProcess(args, 0, json.dumps({'transport': entry}), '')
        if 'remove' in args:
            return subprocess.CompletedProcess(args, 0, '', '')
        if agent == 'claude' and len([c for c in calls if 'add' in c or 'add-json' in c]) == 1:
            return subprocess.CompletedProcess(args, 1, '', 'exists')
        return subprocess.CompletedProcess(args, 0, '', '')
    monkeypatch.setattr(cli_agents.subprocess, 'run', run)
    assert cli_agents.connect_agent(agent, {}, io.StringIO(), io.StringIO()) == 0
    assert any('/restricted' in str(c) and 'new-server' in str(c) for c in calls)
    entry['env']['API_TOKEN'] = 'super-secret-value'
    calls.clear()
    err = io.StringIO()
    assert cli_agents.connect_agent(agent, {}, io.StringIO(), err) != 0
    assert not any('remove' in c for c in calls)
    assert 'super-secret-value' not in str(calls) + err.getvalue()


def test_claude_refuses_opaque_entry_fields_that_could_contain_secrets(monkeypatch):
    entry = {'command': 'old-server', 'args': ['mcp'], 'headers': {'Authorization': 'opaque-secret'}}
    monkeypatch.setattr(cli_agents.shutil, 'which', lambda *a, **k: 'claude')
    monkeypatch.setattr(cli_agents, '_claude_entry', lambda *_: entry)
    calls = []
    monkeypatch.setattr(cli_agents.subprocess, 'run', lambda args, **kwargs: calls.append(args))
    err = io.StringIO()
    assert cli_agents.connect_claude({}, io.StringIO(), err) != 0
    assert calls == []
    assert 'opaque-secret' not in err.getvalue()


@pytest.mark.parametrize('agent', ['codex', 'claude'])
def test_reconnect_refuses_opaque_arguments(monkeypatch, agent):
    entry = {'command': 'old-server', 'args': ['--token', 'opaque-secret']}
    calls = []
    monkeypatch.setattr(cli_agents.shutil, 'which', lambda *a, **k: agent)
    monkeypatch.setattr(cli_agents, '_claude_entry', lambda *_: entry)
    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, json.dumps({'transport': entry}), '')
    monkeypatch.setattr(cli_agents.subprocess, 'run', run)
    err = io.StringIO()
    assert cli_agents.connect_agent(agent, {}, io.StringIO(), err) != 0
    assert not any('remove' in call for call in calls)
    assert 'opaque-secret' not in str(calls) + err.getvalue()
