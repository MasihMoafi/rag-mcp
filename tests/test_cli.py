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
