import pytest
from rag.context_window import expand_passage_windows

TEXT = "Before.\nMiddle.\nAfter."

def record(start, end, path="notes.txt"):
    return {"id": f"{path}:{start}", "file_path": path, "content": TEXT[start:end],
            "original_content": TEXT[start:end], "start_offset": start, "end_offset": end,
            "location": "L2", "score": 0.75}

DOCS = [record(0, 7), record(8, 15), record(16, 22)]

def expand(radius=1, docs=DOCS, source=TEXT, match=None):
    return expand_passage_windows([match or DOCS[1]], docs, lambda path: source, radius, {"notes.txt"})[0]


def test_window_adds_exact_neighbours_preserving_order_score_and_hit():
    result = expand()
    assert result["content"] == result["original_content"] == TEXT
    assert (result["start_offset"], result["end_offset"], result["location"]) == (0, len(TEXT), "L1-L3")
    assert result["id"] == DOCS[1]["id"] and result["score"] == 0.75
    assert DOCS[1]["content"] == "Middle."


def test_off_does_not_read_sources_or_metadata():
    def forbidden(*args):
        pytest.fail("off must not read a source")
    matches = [DOCS[1]]
    assert expand_passage_windows(matches, [], forbidden, 0, set()) is matches


@pytest.mark.parametrize("radius", [-1, 4, True, 1.5])
def test_rejects_invalid_radius(radius):
    with pytest.raises(ValueError):
        expand(radius)


def test_clamps_window_at_source_edges_and_retains_each_hit():
    results = expand_passage_windows([DOCS[0], DOCS[2]], DOCS, lambda path: TEXT, 3, {"notes.txt"})
    assert len(results) == 2 and all(result["content"] == TEXT for result in results)


def test_overlaps_are_sliced_once_and_boundaries_are_exact():
    docs = [record(0, 12), record(8, 22)]
    assert expand(docs=docs, match=docs[1])["content"] == TEXT


@pytest.mark.parametrize("source", ["changed" + TEXT, TEXT.replace("After", "Other")])
def test_stale_source_retains_original_passage(source):
    assert expand(source=source) == DOCS[1]


def test_never_reads_an_excluded_source():
    def forbidden(path):
        pytest.fail("excluded source read")
    match = record(8, 15, "excluded.txt")
    assert expand_passage_windows([match], [match], forbidden, 1, {"notes.txt"}) == [match]


def test_unlocated_or_ambiguous_hit_retains_original():
    assert expand(match={**DOCS[1], "start_offset": None})["start_offset"] is None
    assert expand(docs=DOCS + [DOCS[1]]) == DOCS[1]


def test_unreadable_source_retains_original():
    def unavailable(path):
        raise OSError("unavailable")
    assert expand_passage_windows([DOCS[1]], DOCS, unavailable, 1, {"notes.txt"}) == [DOCS[1]]


def test_window_larger_than_bound_falls_back_to_match():
    text = "x" * 17000
    docs = [{**DOCS[0], "start_offset": 0, "end_offset": 16000, "original_content": text[:16000]},
            {**DOCS[1], "start_offset": 16000, "end_offset": 17000, "original_content": text[16000:]}]
    assert expand_passage_windows([docs[1]], docs, lambda path: text, 1, {"notes.txt"}) == [docs[1]]


@pytest.mark.parametrize('path, location, page', [('notes.pdf', 'p.1-2', 1), ('notes.ipynb', 'cell 1-2', 0)])
def test_page_boundaries_preserve_offsets_and_recompute_location(path, location, page):
    source = 'First page.\fSecond page.'
    docs = [{**record(start, end, path), 'content': source[start:end].replace('\f', ' '),
             'original_content': source[start:end].replace('\f', ' ')}
            for start, end in [(0, 11), (12, len(source))]]
    result, = expand_passage_windows([docs[1]], docs, lambda _: source, 1, {path})
    assert result['content'] == source.replace('\f', ' ')
    assert result['end_offset'] == len(source) and result['location'] == location
    assert result['page_number'] == page and result['start_line'] == 0


def test_source_is_read_once_for_several_matches():
    reads = []
    def reader(path):
        reads.append(path)
        return TEXT
    expand_passage_windows(DOCS, DOCS, reader, 1, {'notes.txt'})
    assert reads == ['notes.txt']


def test_missing_cached_pipeline_does_not_create_an_index(monkeypatch):
    from rag import fetch
    monkeypatch.setattr(fetch, '_rag_system_v2_instances', {})
    def forbidden(*args, **kwargs):
        pytest.fail('answer expansion must not build an index')
    monkeypatch.setattr(fetch, 'get_rag_pipeline_v2', forbidden)
    passages = [DOCS[1]]
    assert fetch.expand_answer_context(passages, 'missing.txt', None, 1) is passages
