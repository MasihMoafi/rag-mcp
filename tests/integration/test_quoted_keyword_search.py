"""Keyword search must answer quoted queries; a whole question in quotes once ended a benchmark run."""

import pytest
from rag.lancedb_backend import LanceDBBackend

DOCUMENTS = [
    {
        "id": "1",
        "content": "The rollout plan needs approvals and a rollback rule.",
        "source": "a.txt",
        "file_path": "a.txt",
    },
    {
        "id": "2",
        "content": "A rule for rollback is listed after the plan needs review.",
        "source": "b.txt",
        "file_path": "b.txt",
    },
    {
        "id": "3",
        "content": "Lunch menus change every Tuesday.",
        "source": "c.txt",
        "file_path": "c.txt",
    },
]


@pytest.fixture
def backend(tmp_path):
    store = LanceDBBackend(
        "quoted", embedding_fn=None, persist_path=str(tmp_path / "db")
    )
    store.add_documents_batch(DOCUMENTS)
    return store


def ids(results):
    return {hit["id"] for hit in results}


def test_a_whole_question_in_quotes_is_searched_by_its_words(backend):
    question = (
        '"What approvals does the rollout plan need, and what is the rollback rule?"'
    )
    assert "1" in ids(backend.search_fts(question))


def test_an_exact_quoted_phrase_finds_only_that_phrase(backend):
    assert ids(backend.search_fts('"rollback rule"')) == {"1"}


@pytest.mark.parametrize(
    "query",
    [
        'the "rollout plan" approvals',
        '"rollout plan" and "rollback rule"',
        '"',
        '""',
        'rollout "plan',
    ],
)
def test_other_quote_shapes_never_raise(backend, query):
    backend.search_fts(query)


def test_quoted_words_that_appear_nowhere_return_nothing(backend):
    assert backend.search_fts('"zebra xylophone"') == []
