from rag.query_decomposition import balanced_passages, decompose_query


def test_explicit_multi_part_question_becomes_focused_facets():
    query = (
        "According to this chapter, why can retrieval metrics look good while the final answer is poor? "
        "Explain the distinct failure points, and give the practical evaluation approach it recommends."
    )

    assert decompose_query(query) == [
        "According to this chapter, why can retrieval metrics look good while the final answer is poor?",
        "Explain the distinct failure points",
        "give the practical evaluation approach it recommends.",
    ]


def test_focused_question_is_not_decomposed():
    query = "What is reciprocal rank fusion?"

    assert decompose_query(query) == [query]


def test_balanced_passages_represents_every_facet_and_keeps_five():
    groups = [
        [{"content": "shared"}, {"content": "a2"}],
        [{"content": "b1"}, {"content": "b2"}],
        [{"content": "c1"}, {"content": "shared"}, {"content": "c3"}],
    ]

    selected = balanced_passages(groups, limit=5)

    assert [item["content"] for item in selected] == ["shared", "b1", "c1", "c3", "b2"]
    assert len(selected) == 5
