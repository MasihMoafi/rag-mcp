import pytest

# These modules replace synthesis with a fixture answer; without a key rag returns passages instead.
MOCKED_SYNTHESIS = {'test_document_refresh', 'test_context_window', 'test_workspace_incremental_index',
                    'test_incremental_index'}


@pytest.fixture(autouse=True)
def _answer_key(request, monkeypatch):
    if request.module.__name__.rsplit('.', 1)[-1] in MOCKED_SYNTHESIS:
        monkeypatch.setenv('OPENROUTER_API_KEY', 'test-only')
