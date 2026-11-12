import pytest


@pytest.fixture(autouse=True)
def _no_codex_sign_in(monkeypatch):
    """Tests never reach the developer's real Codex account; a test that needs it fakes the sign-in."""
    from rag import codex_answers

    monkeypatch.setattr(codex_answers, 'signed_in', lambda: False)
