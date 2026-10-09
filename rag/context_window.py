"""Exact-source neighbouring windows for answer evidence; ranking stays on small chunks."""
from collections import defaultdict
from pathlib import Path

from .ingestion import locate_chunk

MAX_WINDOW_CHARACTERS = 16000


def expand_passage_windows(passages, documents, read_source, radius, allowed_paths):
    if type(radius) is not int or not 0 <= radius <= 3:
        raise ValueError('Neighbour context must be a whole number from 0 to 3.')
    if radius == 0:
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
    for passage in passages:
        expanded.append(passage)
        path = passage.get('file_path')
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
        window = rows[max(0, index - radius):index + radius + 1]
        start = min(row['start_offset'] for row in window)
        end = max(row['end_offset'] for row in window)
        if end - start > MAX_WINDOW_CHARACTERS:
            continue
        if path not in source_cache:
            try:
                source_cache[path] = read_source(path)
            except (OSError, ValueError, RuntimeError):
                source_cache[path] = None
        source = source_cache[path]
        if not isinstance(source, str) or any(
            row['end_offset'] > len(source)
            or source[row['start_offset']:row['end_offset']].replace('\f', ' ') !=
               (row.get('original_content') or row['content'])
            for row in window
        ):
            continue
        raw = source[start:end]
        text = raw.replace('\f', ' ')
        location, _ = locate_chunk(None, source, raw, Path(path).suffix.lower(), start)
        first_line = source.count('\n', 0, start) + 1
        last_line = first_line + raw.count('\n')
        expanded[-1] = {**passage, 'content': text, 'original_content': text,
                        'start_offset': start, 'end_offset': end, 'location': location,
                        'start_line': first_line if location.startswith('L') else 0,
                        'end_line': last_line if location.startswith('L') else 0,
                        'page_number': source.count('\f', 0, start) + 1 if location.startswith('p.') else 0}
    return expanded
