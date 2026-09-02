import io
import json
from pathlib import Path

from rag.cli import main


def test_doctor_reports_codex_login_as_json(tmp_path: Path):
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nprintf 'Logged in using ChatGPT\\n'\n")
    codex.chmod(0o755)
    output = io.StringIO()

    code = main(
        ["doctor", "--json"],
        stdout=output,
        environ={"PATH": str(tmp_path)},
    )

    report = json.loads(output.getvalue())
    assert code == 0
    assert report["codex"] == {
        "available": True,
        "authenticated": True,
        "method": "ChatGPT",
    }


def test_query_prints_the_real_query_result_contract(tmp_path: Path):
    document = tmp_path / "facts.md"
    document.write_text("The launch codename is Cedar Lantern.")
    output = io.StringIO()

    code = main(
        ["query", "What is the launch codename?", "--path", str(document)],
        stdout=output,
        query_fn=lambda query, path: json.dumps(
            {"result": [{"content": document.read_text(), "source": path}]}
        ),
    )

    assert code == 0
    assert json.loads(output.getvalue()) == {
        "result": [
            {
                "content": "The launch codename is Cedar Lantern.",
                "source": str(document),
            }
        ]
    }


def test_doctor_reports_missing_and_signed_out_codex(tmp_path: Path):
    output = io.StringIO()
    assert main(["doctor", "--json"], stdout=output, environ={"PATH": str(tmp_path)}) == 1
    assert json.loads(output.getvalue())["codex"] == {
        "available": False,
        "authenticated": False,
        "method": None,
    }

    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nprintf 'Not logged in\n'\nexit 1\n")
    codex.chmod(0o755)
    output = io.StringIO()
    assert main(["doctor", "--json"], stdout=output, environ={"PATH": str(tmp_path)}) == 1
    assert json.loads(output.getvalue())["codex"] == {
        "available": True,
        "authenticated": False,
        "method": None,
    }


def test_connect_codex_registers_the_current_server(tmp_path: Path):
    calls = tmp_path / "calls"
    codex = tmp_path / "codex"
    codex.write_text(
        "#!/bin/sh\n"
        "if [ \"$1 $2\" = \"mcp get\" ]; then exit 1; fi\n"
        f"printf '%s\\n' \"$@\" > '{calls}'\n"
    )
    codex.chmod(0o755)
    output = io.StringIO()

    code = main(
        ["connect-codex"],
        stdout=output,
        environ={"PATH": str(tmp_path)},
    )

    assert code == 0
    args = calls.read_text().splitlines()
    assert args[:4] == ["mcp", "add", "rag", "--"]
    assert Path(args[4]).resolve() == Path(__import__("sys").executable).resolve()
    assert Path(args[5]).resolve().name == "server.py"
