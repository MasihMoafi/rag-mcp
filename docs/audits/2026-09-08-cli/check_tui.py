"""Archived PTY check for the earlier preset menu; superseded by check_tmux.py."""
import fcntl
import json
import os
from pathlib import Path
import pty
import re
import select
import struct
import subprocess
import tempfile
import termios
import time

root = Path(tempfile.mkdtemp(prefix="rag-tui-"))
(root / "notes.txt").write_text("Project deadline is Friday.\n")
env = dict(os.environ, RAG_MCP_CONFIG=str(root / "config.json"),
           RAG_MCP_DATA_DIR=str(root / "index"), TERM="xterm-256color",
           PROMPT_TOOLKIT_NO_CPR="1")
master, slave = pty.openpty()
fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 32, 110, 0, 0))
process = subprocess.Popen(["rag"], cwd=root, env=env, stdin=slave,
                           stdout=slave, stderr=slave, close_fds=True)
os.close(slave)
transcript = bytearray()
pending = bytearray()


def expect(needle, timeout=30):
    deadline = time.monotonic() + timeout
    while needle not in re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", pending.decode(errors="replace")):
        if time.monotonic() > deadline:
            raise RuntimeError(f"Timed out: {needle}\n{pending.decode(errors='replace')[-1600:]}")
        if select.select([master], [], [], 0.2)[0]:
            chunk = os.read(master, 65536)
            transcript.extend(chunk)
            pending.extend(chunk)
    pending.clear()
    time.sleep(0.15)


def keys(value):
    os.write(master, value.encode())


down, up, enter, escape = "\x1b[B", "\x1b[A", "\r", "\x1b"
try:
    expect("Quit")
    keys(enter)
    expect("already installed")
    keys(enter)
    expect("Other · type a model or server")
    keys(enter)
    expect("Connect a coding agent?", 60)
    assert json.loads((root / "config.json").read_text())["embed_provider"] == "ollama"
    keys(enter)
    expect("Quit")
    keys(down * 3 + enter)
    expect("Change setting")
    keys(enter)
    expect("Type your own")
    keys(up * 2 + enter)
    expect("Quit")
    assert json.loads((root / "config.json").read_text())["top_k"] == "1"
    keys(down + enter)
    expect("Type a path")
    keys(enter)
    expect("What would you like to find?")
    keys(up + enter)
    expect("? What would you like to find?")
    keys(escape)
    expect("Type your own")
    keys(escape)
    expect("Quit")
    keys(down * 2 + enter)
    expect("Coding agent")
    keys(escape)
    expect("Quit")
    keys(up + enter)
    process.wait(timeout=5)
    assert process.returncode == 0
    assert b"Traceback" not in transcript
    print("PASS: global launch, arrow-only Ollama setup and live probe, skip connection, "
          "save result count, folder picker, custom question entry and Escape, "
          "agent picker cancellation, return to menu, quit.")
finally:
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=5)
    os.close(master)
    target = Path(__file__).with_name("tui-terminal.log")
    target.write_bytes(transcript)
    print(f"Terminal evidence: {target}")
