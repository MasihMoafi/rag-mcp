"""Extract static HTML content while retaining physical source-line positions."""

from html import unescape
from html.parser import HTMLParser
import re

from .text import extract_plain_text

_VOID = set('area base br col embed hr img input link meta param source track wbr'.split())
_SKIP = {'script', 'style', 'nav', 'template'}
_BLOCK = set('p div section article header footer h1 h2 h3 h4 h5 h6 li ul ol table tr td th br hr'.split())
# A page still carrying template placeholders is scaffolding, not content.
_PLACEHOLDER = re.compile(r'\{\{.*?\}\}|\{%.*?%\}')


class _ContentReader(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.lines = [[] for _ in source.split('\n')]
        self.stack = []
        self.pre_lines = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        hidden = (bool(self.stack and self.stack[-1][1]) or tag in _SKIP
                  or 'hidden' in attrs or str(attrs.get('aria-hidden', '')).lower() == 'true')
        if tag in _BLOCK and not hidden:
            self.handle_data(' ')
        if tag not in _VOID:
            self.stack.append((tag, hidden))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break
        if tag in _BLOCK:
            self.handle_data(' ')

    def handle_data(self, data):
        if self.stack and self.stack[-1][1]:
            return
        first = self.getpos()[0] - 1
        preformatted = any(tag == 'pre' for tag, _ in self.stack)
        for offset, part in enumerate(data.split('\n')):
            self.lines[first + offset].append(part)
            if preformatted:
                self.pre_lines.add(first + offset)

    def handle_entityref(self, name):
        self.handle_data(unescape('&' + name + ';').replace('\n', ' '))

    def handle_charref(self, name):
        self.handle_entityref('#' + name)


def extract_html_text(file_path):
    source = extract_plain_text(file_path)
    parser = _ContentReader(source)
    parser.feed(source)
    parser.close()
    lines = [''.join(parts) if index in parser.pre_lines
             else re.sub(r'\s+', ' ', ''.join(parts)).strip()
             for index, parts in enumerate(parser.lines)]
    if any(_PLACEHOLDER.search(line) for index, line in enumerate(lines)
           if index not in parser.pre_lines):
        return ''
    return '\n'.join(lines)
