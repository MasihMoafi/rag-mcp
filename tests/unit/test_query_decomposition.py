from rag.query_decomposition import balanced_passages, decompose_query
import pytest


@pytest.mark.parametrize(('query', 'subject'), [
    ('How can we assess solar-panel efficiency? Recommend a practical installation plan.', 'solar-panel efficiency'),
    ('How can we measure river pollution? Explain a practical cleanup approach.', 'river pollution'),
])
def test_off_domain_evaluation_keeps_subject_in_each_facet(query, subject):
    facets = decompose_query(query)
    assert len(facets) >= 2
    assert all(subject in facet for facet in facets)
    assert all('RAG' not in facet and 'retrieval' not in facet for facet in facets)


def test_explicit_multi_part_question_becomes_focused_facets():
    query = (
        "According to this chapter, why can retrieval metrics look good while the final answer is poor? "
        "Explain the distinct failure points, and give the practical evaluation approach it recommends."
    )

    facets = decompose_query(query)
    assert len(facets) == 3
    assert facets[0].startswith("According to this chapter, why can retrieval metrics look good while the final answer is poor?")
    assert facets[1].startswith("Explain the distinct failure points")
    assert facets[2].startswith("give the practical evaluation approach it recommends.")
    assert all(query in facet for facet in facets)


def test_focused_question_is_not_decomposed():
    query = "What is reciprocal rank fusion?"

    assert decompose_query(query) == [query]


def test_book_answer_instructions_do_not_become_search_facets():
    question = "Which passages connect sustained work, preparation, or attention to success?"
    query = (
        question + "\nReturn two useful passages from this PDF. "
        "For each, give the exact one-based PDF page number, the printed page if visible, "
        "the section, a verbatim anchor of at most ten words, a brief paraphrase, "
        "and a context caveat. State when a page number or passage text is unavailable. "
        "Do not invent references."
    )

    assert decompose_query(query) == [question]


def test_answer_instructions_do_not_remove_a_second_content_question():
    query = "Which database is used? Explain the migration approach. Return the answer as JSON."

    assert decompose_query(query) == [
        "Which database is used?", "Explain the migration approach."
    ]


def test_instruction_only_query_keeps_its_search_text():
    query = "Return two passages from this book."

    assert decompose_query(query) == [query]


def test_return_can_request_content_instead_of_an_answer_format():
    query = "Which country is discussed? Return its history."
    facets = decompose_query(query)
    assert facets[0] == query
    assert any(facet.startswith("Return its history.") and query in facet for facet in facets)


def test_alternative_evidence_topics_have_separate_searches():
    query = 'Which passages show learning, independent thinking, or disciplined judgment?'

    assert decompose_query(query) == [
        'learning', 'independent thinking', 'disciplined judgment',
    ]


def test_or_in_an_ordinary_question_does_not_change_its_scope():
    query = 'Was the database SQLite or PostgreSQL?'

    assert decompose_query(query) == [query]


def test_unrelated_multipart_question_is_not_expanded_with_evaluation_terms():
    query = "Which database is used? Explain the practical migration approach."

    assert decompose_query(query) == [
        "Which database is used?",
        "Explain the practical migration approach.",
    ]


def test_balanced_passages_represents_every_facet_and_keeps_five():
    groups = [
        [{"content": "shared"}, {"content": "a2"}],
        [{"content": "b1"}, {"content": "b2"}],
        [{"content": "c1"}, {"content": "shared"}, {"content": "c3"}],
    ]

    selected = balanced_passages(groups, limit=5)

    assert [item["content"] for item in selected] == ["shared", "b1", "c1", "c3", "b2"]
    assert len(selected) == 5


@pytest.mark.parametrize('query', [
    'Which storage engines have reported errors, and what are the corresponding ticket IDs?',
    'Which services have an exception to the network policy, and how long does each exception last?',
    'Which database uses signed tokens? How are they rotated?',
    'The transport standard disallows caching, but ADR-7 describes a cache. What is the actual policy, and how should operators describe outage behavior?',
])
def test_dependent_facets_keep_original_query_and_subject(query):
    facets = decompose_query(query)
    assert facets[0] == query
    assert 1 < len(facets) <= 3
    assert all(query in facet for facet in facets)


def test_independent_questions_stay_separate():
    query = 'Which database is used? Where is the backup stored?'
    assert decompose_query(query) == ['Which database is used?', 'Where is the backup stored?']


def test_unrelated_background_does_not_merge_independent_questions():
    query = 'The database runs on Linux. What is the backup schedule? Where is Paris?'
    assert decompose_query(query) == [
        'The database runs on Linux.', 'What is the backup schedule?', 'Where is Paris?',
    ]


def test_dependent_facets_respect_query_and_passage_budgets():
    query = 'Which service owns the key, and how is it rotated, and where is its backup?'
    facets = decompose_query(query, max_facets=2)
    assert len(facets) == 2 and facets[0] == query
    assert all(query in facet for facet in facets)
    groups = [[{'id': 'shared', 'content': 'shared'}, {'id': str(i), 'content': str(i)}]
              for i in range(len(facets))]
    selected = balanced_passages(groups, limit=2)
    assert len(selected) == 2
    assert len({p['id'] for p in selected}) == 2
