from rag.query_decomposition import balanced_passages, decompose_query


def test_explicit_multi_part_question_becomes_focused_facets():
    query = (
        "According to this chapter, why can retrieval metrics look good while the final answer is poor? "
        "Explain the distinct failure points, and give the practical evaluation approach it recommends."
    )

    assert decompose_query(query) == [
        (
            "Failure points across document ingestion (including outdated or improperly parsed "
            "documents), flawed retrieval, and hallucinated or incomplete answer generation: "
            "According to this chapter, why can retrieval metrics look good while the final answer "
            "is poor? Explain the distinct failure points"
        ),
        (
            "Recommended practical offline evaluation lifecycle: fixed golden dataset, CI/CD "
            "evaluation gates, regression thresholds, and tuning."
        ),
        (
            "Recommended practical online evaluation lifecycle: asynchronous sampled live "
            "traffic, monitoring, A/B testing, and human feedback."
        ),
    ]


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

    assert decompose_query(query) == ["Which country is discussed?", "Return its history."]


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
