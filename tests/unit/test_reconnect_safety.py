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
            registry["rag"] = [entry["command"], *entry["args"]]
            return subprocess.CompletedProcess(args, 0, "", "")
        command = "new-server" if "new-server" in args else "old-server"
        if "rag" in registry:
            return subprocess.CompletedProcess(args, 1, "", "rag already exists")
        if command == "new-server" and reject_new:
            return subprocess.CompletedProcess(args, 1, "", "add failed")
        registry["rag"] = [command, "mcp"]
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(cli_agents.subprocess, "run", run)
    return registry, calls


@pytest.mark.parametrize("agent", AGENTS)
def test_failed_reconnect_keeps_the_old_registration(monkeypatch, agent):
    registry, _ = fake_client(monkeypatch, agent, reject_new=True)
    assert cli_agents.connect_agent(agent, {}, io.StringIO(), io.StringIO()) != 0
    assert registry["rag"] == ["old-server", "mcp"]


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
