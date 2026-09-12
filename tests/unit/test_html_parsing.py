"""HTML content extraction retains physical source lines for citations."""

from rag.parsers.html import extract_html_text


def test_html_removes_noncontent_and_preserves_source_lines(tmp_path):
    source = tmp_path / 'page.html'
    source.write_text('<html>\n<nav>Navigation junk</nav>\n<script>secret = 42</script>\n'
                      '<style>.junk {color:red}</style>\n<h1>Cedar &amp; Birch</h1>\n'
                      '<p>Use <strong>two</strong> workers.</p>\n'
                      '<div hidden><p>Hidden junk</p></div>\n</html>')
    lines = extract_html_text(str(source)).split('\n')
    assert len(lines) == 8
    assert lines[4] == 'Cedar & Birch'
    assert lines[5] == 'Use two workers.'
    assert not any(word in '\n'.join(lines) for word in ['junk', 'secret', '<strong>'])


def test_html_retains_code_tables_and_non_english_text(tmp_path):
    source = tmp_path / 'page.html'
    source.write_text('<pre>def run():\n    return 42</pre>\n'
                      '<table><tr><th>نام</th><td>Cedar</td></tr></table>\n'
                      '<!-- junk --><p>یک متن &lt;sample&gt;</p>')
    text = extract_html_text(str(source))
    assert 'def run():' in text and 'return 42' in text
    assert 'نام Cedar' in text and 'یک متن <sample>' in text
    assert 'junk' not in text


def test_hidden_void_element_does_not_hide_following_content(tmp_path):
    source = tmp_path / 'page.html'
    source.write_text('<input hidden><p>Visible evidence</p><div hidden>Invisible</div>')
    assert extract_html_text(str(source)) == 'Visible evidence'


def test_entities_do_not_split_words_or_invent_source_lines(tmp_path):
    source = tmp_path / 'page.html'
    source.write_text('<p>caf&eacute; &NewLine; &#169;</p>\n<p>Second line</p>')
    assert extract_html_text(str(source)) == 'café ©\nSecond line'
