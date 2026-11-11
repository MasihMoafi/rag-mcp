"""Exact-source answer evidence: neighbouring windows, or a short source whole; ranking stays on small chunks."""
from collections import defaultdict
from pathlib import Path

from .ingestion import locate_chunk

MAX_WINDOW_CHARACTERS = 16000


def _span(passage, source, path, start, end):
    """The passage widened to source[start:end], with its location recomputed."""
    raw = source[start:end]
    text = raw.replace('\f', ' ')
    location, _ = locate_chunk(None, source, raw, Path(path).suffix.lower(), start)
    first_line = source.count('\n', 0, start) + 1
    last_line = first_line + raw.count('\n')
    return {**passage, 'content': text, 'original_content': text,
            'start_offset': start, 'end_offset': end, 'location': location,
            'start_line': first_line if location.startswith('L') else 0,
            'end_line': last_line if location.startswith('L') else 0,
            'page_number': source.count('\f', 0, start) + 1 if location.startswith('p.') else 0}


def expand_passage_windows(passages, documents, read_source, radius, allowed_paths, whole_chars=0):
    """Widen each verified passage to its neighbours (`radius` each side).

    A source of at most `whole_chars` characters replaces its passages instead: whole, once, at
    its best rank (EnterpriseRAG-Bench, 11 Nov 2026: 75.2% -> 81.0% correct). Unverifiable
    passages stay as they are.
    """
    if type(radius) is not int or not 0 <= radius <= 3:
        raise ValueError('Neighbour context must be a whole number from 0 to 3.')
    if type(whole_chars) is not int or whole_chars < 0:
        raise ValueError('Whole-document size must be a whole number of characters, 0 or more.')
    if radius == 0 and whole_chars == 0:
        return passages
    by_source = defaultdict(list)
    for document in documents:
        start, end = document.get('start_offset'), document.get('end_offset')
        if type(start) is int and type(end) is int and 0 <= start < end:
            by_source[document.get('file_path')].append(document)
    for rows in by_source.values():
        rows.sort(key=lambda row: (row['start_offset'], row['end_offset']))
    source_cache = {}
    expanded = []
    whole_sources = set()

    def source_text(path):
        if path not in source_cache:
            try:
                source_cache[path] = read_source(path)
            except (OSError, ValueError, RuntimeError):
                source_cache[path] = None
        source = source_cache[path]
        return source if isinstance(source, str) else None

    def verified(rows, source):
        return source is not None and all(
            row['end_offset'] <= len(source)
            and source[row['start_offset']:row['end_offset']].replace('\f', ' ') ==
                (row.get('original_content') or row['content'])
            for row in rows)

    for passage in passages:
        path = passage.get('file_path')
        if path in whole_sources:
            continue  # Already given whole at a better rank.
        expanded.append(passage)
        if path not in allowed_paths:
            continue
        rows = by_source.get(path, [])
        matches = [i for i, row in enumerate(rows)
                   if row.get('id') == passage.get('id')
                   and row['start_offset'] == passage.get('start_offset')
                   and row['end_offset'] == passage.get('end_offset')
                   and (row.get('original_content') or row['content']) ==
                       (passage.get('original_content') or passage.get('content'))]
        if len(matches) != 1:
            continue
        index = matches[0]
        if whole_chars:
            source = source_text(path)
            if source is not None and len(source) <= whole_chars and verified([rows[index]], source):
                expanded[-1] = dict(_span(passage, source, path, 0, len(source)), whole_document=True)
                whole_sources.add(path)
                continue
        if radius == 0:
            continue
        window = rows[max(0, index - radius):index + radius + 1]
        start = min(row['start_offset'] for row in window)
        end = max(row['end_offset'] for row in window)
        if end - start > MAX_WINDOW_CHARACTERS:
            continue
        source = source_text(path)
        if not verified(window, source):
            continue
        expanded[-1] = _span(passage, source, path, start, end)
    return expanded
