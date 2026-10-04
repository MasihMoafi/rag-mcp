import json

import pytest

from rag.passage_answers import ABSTENTION, render_selections, requests_passages, selection_limits


def _selection(anchor, **changes):
    return json.dumps({'selections': [{'passage': 1, 'anchor': anchor,
                       'paraphrase': 'The speaker advises study.', 'caveat': 'Advice, not proof.',
                       'addresses_request': True, **changes}]})


def test_reference_request_is_not_always_a_passage_selection():
    assert requests_passages('Which passages show learning?')
    assert not requests_passages('What year did this happen? Give its PDF page number.')


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
