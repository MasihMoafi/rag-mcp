"""UTF-8 source reader for formats without a specialized extractor."""

import re

_FENCE = re.compile(r"(^(?:```|~~~)[^\n]*\n[\s\S]*?^(?:```|~~~)[ \t]*$)", re.M)
_SETEXT = re.compile(r"^([^\s|#>-].*)\n[ \t]*(={3,}|-{3,})[ \t]*$", re.M)
_RULE_LINE = re.compile(r"^[ \t]*#*[ \t]*([-=_*])(?:[ \t]*\1){2,}[ \t]*(?:\n|$)", re.M)
_TABLE_RULE = re.compile(r"^[ \t]*\|?(?:[ \t]*:?-{3,}:?[ \t]*\|)+(?:[ \t]*:?-{3,}:?)?[ \t]*(?:\n|$)", re.M)
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~])")
_LABELLED_TAG = re.compile(r'<[a-z][^>]*\b(?:alt|aria-label)="([^"]*)"[^>]*>', re.I)
_TAG = re.compile(r"</?(?:span|div|p|br|hr|sup|sub|font|a|img|u|small|center|figure|figcaption|em|strong|b|i)\b[^>]*>", re.I)


def extract_plain_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as source:
        return source.read()


def clean_text(text: str, markdown: bool) -> str:
    """Remove layout marks converters leave in text (rules, table separators, escapes, padding).

    Fenced code blocks are left exactly as written. Studio's browser parser applies the same rules.
    """
    if markdown:
        text = re.sub(r"\A---\n([\s\S]*?)\n---\n", r"\1\n", text)
    parts = _FENCE.split(text)
    for index in range(0, len(parts), 2):
        part = parts[index]
        if markdown:
            part = _SETEXT.sub(lambda match: f"{'#' if match[2][0] == '=' else '##'} {match[1]}", part)
        part = _TABLE_RULE.sub("", _RULE_LINE.sub("", part))
        part = re.sub(r"([-=_])\1{3,}", " ", part)
        if markdown:
            part = _TAG.sub("", _LABELLED_TAG.sub(r"\1", _ESCAPE.sub(r"\1", part)))
        part = re.sub(r"[ \t]+$", "", part, flags=re.M)
        part = re.sub(r"(\S)[ \t]{2,}(?=\S)", r"\1 ", part)
        parts[index] = re.sub(r"\n{3,}", "\n\n", part)
    return "".join(parts).strip()
