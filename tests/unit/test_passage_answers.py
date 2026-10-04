import json

import pytest

from rag.passage_answers import (ABSTENTION, render_selections, requests_passages, requests_references,
                                 selection_limits)


def _selection(anchor, **changes):
    return json.dumps({'selections': [{'passage': 1, 'anchor': anchor,
                       'paraphrase': 'The speaker advises study.', 'caveat': 'Advice, not proof.',
                       'addresses_request': True, **changes}]})


def test_reference_request_is_not_always_a_passage_selection():
    assert requests_passages('Which passages show learning?')
    assert not requests_passages('What year did this happen? Give its PDF page number.')


@pytest.mark.parametrize('query', [
    'Which passages show learning?',
    'Return two useful passages. Use an anchor of at most ten words.',
    'Find 2 short passages that mention the retreat.',
    'Give me three passages about courage.',
    'Which of the passages warn against haste?',
])
def test_a_request_for_passages_selects_passages(query):
    assert requests_passages(query)


@pytest.mark.parametrize('query', [
    'Which function returns the passages to synthesis?',
    'Which code returns passages to the reranker?',
    'Find where the server trims passages before synthesis.',
])
def test_a_question_that_only_mentions_passages_is_answered_normally(query):
    assert not requests_passages(query)


@pytest.mark.parametrize('query', [
    'Give a short quote and its exact PDF page number.',
    'Give the PDF page number.',
    'Cite the sources.',
    'Include the source references.',
    'Which passages show learning?',
])
def test_a_request_for_references_gets_them(query):
    assert requests_references(query)


@pytest.mark.parametrize('query', [
    'Which function re-reads only the changed source files?',
    'When does the code restore PDF page breaks from the source offsets?',
    'What error does a folder with more than 500 source files cause?',
    'What is the largest number of passages a request can select, and what is the default?',
    'Which function builds the continuity prompt from the memory sources?',
    'How does the parser handle quotes inside strings?',
    'List the source files that the indexer skips.',
])
def test_a_question_that_only_mentions_sources_gets_no_references(query):
    assert not requests_references(query)


def test_quote_page_and_word_limit_come_from_source_evidence():
    quote = 'One two three four five six seven eight nine ten eleven.'
    result = render_selections(_selection(quote), [{'content': quote}],
        ['Printed page number: unavailable.\n[PDF page 19]\nEarlier.\n[PDF page 20]\n' + quote],
        cite_sources=True, max_anchor_words=8)
    assert 'PDF page 20.' in result
    assert '“One two three four five six seven eight”' in result
    assert 'nine' not in result
    assert '[1]' in result


def test_empty_selections_abstain():
    assert render_selections('{"selections": []}', [], [], cite_sources=False) == ABSTENTION


def test_supported_quote_that_does_not_address_request_is_excluded():
    result = render_selections(_selection('Evidence', addresses_request=False),
        [{'content': 'Evidence'}], ['Evidence'], cite_sources=False)
    assert result == ABSTENTION


@pytest.mark.parametrize('answer', [
    'Not JSON.', '{"selections": "bad"}', _selection('Invented quote'),
    _selection('Evidence', passage=9), _selection('Evidence', passage=True),
])
def test_invalid_or_invented_quote_is_an_error(answer):
    with pytest.raises(RuntimeError, match='unsupported quoted text'):
        render_selections(answer, [{'content': 'Evidence'}], ['Evidence'], cite_sources=False)


def test_missing_page_data_stays_unavailable():
    result = render_selections(_selection('Evidence'), [{'content': 'Evidence'}],
        ['Evidence'], cite_sources=False)
    assert 'Exact PDF page: unavailable' in result


def test_requested_counts_and_shorter_anchor_limits():
    assert selection_limits('Return two useful passages. Use an anchor of at most ten words.') == (2, 10)
    assert selection_limits('Give one passage. Use at most three words.') == (1, 3)
    assert selection_limits('Return 4 passages. Use up to 5 words.') == (4, 5)


def test_unique_quote_can_correct_a_model_passage_number():
    answer = _selection('The speaker\'s advice.')
    result = render_selections(answer, [
        {'content': 'Unrelated text.'}, {'content': 'The speaker’s advice.'}],
        ['[PDF page 1]\nUnrelated text.', '[PDF page 2]\nThe speaker’s advice.'],
        cite_sources=True)
    assert 'PDF page 2.' in result
    assert '“The speaker’s advice.”' in result
    assert '[2]' in result


def test_invalid_quote_does_not_discard_a_verified_selection():
    answer = json.loads(_selection('A verified quote.'))
    answer['selections'].extend(json.loads(_selection('A changed quote.'))['selections'])
    result = render_selections(json.dumps(answer), [{'content': 'A verified quote.'}],
        ['[PDF page 7]\nA verified quote.'], cite_sources=False)
    assert '“A verified quote.”' in result
    assert 'PDF page 7.' in result
    assert 'A changed quote.' not in result
    assert 'only 1 selection' in result
