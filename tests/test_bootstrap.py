from __future__ import annotations

import importlib.util
from pathlib import Path


_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap.py"
_SPEC = importlib.util.spec_from_file_location("bootstrap_script", _BOOTSTRAP_PATH)
bootstrap = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(bootstrap)


def test_check_uv_raises_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _name: None)
    try:
        bootstrap._check_uv()
        assert False, "expected _check_uv to raise when uv is missing"
    except RuntimeError as exc:
        assert "Missing required tool: uv" in str(exc)


def test_venv_python_resolves_posix_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bootstrap.os, "name", "posix")
    assert bootstrap._venv_python(tmp_path) == tmp_path / ".venv" / "bin" / "python"
