"""Check quoted anchors and derive their locations from retrieved evidence."""

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


ABSTENTION = "I don't have enough information to answer that question."
_NUMBERS = dict(zip(('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'), range(1, 11)))


class PassageAnswerError(RuntimeError):
    """The answer contains an invalid or unsupported passage selection."""


def selection_limits(query: str) -> tuple[int, int]:
    number = r'(\d+|' + '|'.join(_NUMBERS) + r')'
    count = re.search(r'\b(?:find|select|return|give|provide)\s+' + number + r'\b[^?.]*\bpassages?\b', query, re.I)
    words = re.search(r'\b(?:at most|up to|no more than)\s+' + number + r'\s+words\b', query, re.I)

    def value(match, default):
        text = match[1].lower() if match else ''
        return int(text) if text.isdigit() else _NUMBERS.get(text, default)

    return max(1, min(5, value(count, 2))), max(1, min(10, value(words, 10)))


def requests_passages(query: str) -> bool:
    return bool(re.search(r'\b(?:which|find|select|return|give|provide)\b[^?.]*\bpassages\b', query, re.I))


def selection_format(max_passages: int) -> dict:
    fields = {'passage': {'type': 'integer'}, 'anchor': {'type': 'string'},
              'paraphrase': {'type': 'string'}, 'caveat': {'type': 'string'},
              'addresses_request': {'type': 'boolean'}}
    return {'type': 'json_schema', 'json_schema': {
        'name': 'passage_selections', 'strict': True, 'schema': {
            'type': 'object', 'properties': {'selections': {
                'type': 'array', 'maxItems': max_passages, 'items': {
                    'type': 'object', 'properties': fields,
                    'required': list(fields), 'additionalProperties': False}}},
            'required': ['selections'], 'additionalProperties': False}}}


def render_selections(answer: str, passages: Sequence[Mapping[str, Any]],
                      evidence: Sequence[str], *, cite_sources: bool,
                      max_passages: int = 2, max_anchor_words: int = 10) -> str:
    try:
        data = json.loads(answer)
        selections = data['selections']
        if set(data) != {'selections'} or not isinstance(selections, list) or len(selections) > max_passages:
            raise ValueError
        rendered = []
        seen = set()
        unsupported_quote = False
        for item in selections:
            if set(item) != {'passage', 'anchor', 'paraphrase', 'caveat', 'addresses_request'}:
                raise ValueError
            if type(item['addresses_request']) is not bool:
                raise ValueError
            if not item['addresses_request']:
                continue
            rank = item['passage']
            if type(rank) is not int or not 1 <= rank <= len(passages):
                raise ValueError
            if any(not isinstance(item[key], str) or not item[key].strip()
                   for key in ('anchor', 'paraphrase', 'caveat')):
                raise ValueError
            anchor = ' '.join(item['anchor'].split())
            sources = [' '.join(str(passage.get('original_content') or passage.get('content') or '').split())
                       for passage in passages]
            punctuation = str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"'})

            def find(source):
                return source.translate(punctuation).lower().find(anchor.translate(punctuation).lower())

            start = find(sources[rank - 1])
            if start < 0:
                matches = [(index + 1, find(source)) for index, source in enumerate(sources) if find(source) >= 0]
                if len(matches) != 1:
                    unsupported_quote = True
                    continue
                rank, start = matches[0]
            anchor = sources[rank - 1][start:start + len(anchor)]
            # A shorter prefix remains a verbatim quote and respects the word cap.
            anchor = ' '.join(anchor.split()[:max_anchor_words])
            if anchor in seen:
                continue
            seen.add(anchor)
            page_parts = re.split(r'\[PDF page ([1-9]\d*)\]\n', evidence[rank - 1])
            pages = [int(page_parts[i]) for i in range(1, len(page_parts), 2)
                     if anchor in ' '.join(page_parts[i + 1].split())]
            if len(set(pages)) == 1:
                location = f'PDF page {pages[0]}'
            else:
                location = 'Exact PDF page: unavailable'
                match = re.search(r'PDF page range: ([0-9]+-[0-9]+)', evidence[rank - 1])
                if match:
                    location += f' (source range {match[1]})'
            citation = f' [{rank}]' if cite_sources else ''
            rendered.append(
                f'{len(rendered) + 1}. {location}. Printed page: unavailable. Section: unavailable.\n'
                f'Anchor: “{anchor}”\nParaphrase: {item["paraphrase"].strip()}{citation}\n'
                f'Context: {item["caveat"].strip()}'
            )
        if not rendered:
            if unsupported_quote:
                raise ValueError
            return ABSTENTION
        result = '\n\n'.join(rendered)
        if len(rendered) < max_passages:
            result += f'\n\nThe supplied evidence supports only {len(rendered)} selection(s).'
        return result
    except (ValueError, KeyError, TypeError, IndexError) as error:
        raise PassageAnswerError('Passage synthesis returned invalid or unsupported quoted text.') from error
