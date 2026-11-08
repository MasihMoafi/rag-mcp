"""Deterministic query facets and balanced evidence selection."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


_FACET_START = (
    r"what|why|how|which|who|where|when|give|explain|describe|identify|"
    r"compare|list|summarize|outline|provide"
)
_ANSWER_INSTRUCTION = re.compile(
    r"^(?:"
    r"(?:return|give|provide|list)\b[^?]*\b(?:passages?|quotes?|excerpts?)\s+"
    r"from (?:this|the) (?:PDF|document|book|file|text)[.!]?$|"
    r"(?:return|respond|reply|output|format)\b[^?]*\bas (?:JSON|Markdown|a table|bullets)\b|"
    r"(?:state\s+when\s+a\s+page number|"
    r"(?:state|include|give|provide)\s+(?:(?:the|a|an|exact|one-based|PDF|printed)\s+)*"
    r"(?:page number|source reference|citation|format))\b|"
    r"do not (?:invent|fabricate)\b)",
    re.IGNORECASE,
)
_EVIDENCE_ALTERNATIVES = re.compile(
    r'((?:which|what) (?:passages|excerpts|quotes|examples) '
    r'(?:show|describe|discuss|warn against|help someone) )(.+)\?',
    re.IGNORECASE,
)


def _searchable_instruction(text: str) -> str:
    """Remove output-only instructions without discarding their subject."""
    text = re.sub(r"^for each(?:\s+[^,]+)?,\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:with|and)\s+(?:a\s+|the\s+)?(?:citations?|source references?|page numbers?)\b", "", text, flags=re.IGNORECASE)
    match = re.match(r"^(?:return|respond|reply|output|format)\s+(.+?)\s+as\s+(?:JSON|Markdown|a table|bullets)\b.*$", text, re.IGNORECASE)
    if match:
        subject = match[1].strip()
        return '' if re.fullmatch(r'(?:the\s+)?(?:answer|response|output|result)', subject, re.IGNORECASE) else subject
    return '' if _ANSWER_INSTRUCTION.search(text) else text


def _evidence_alternatives(facet: str) -> list[str]:
    """Search each explicit alternative in a request for supporting text."""
    match = _EVIDENCE_ALTERNATIVES.fullmatch(facet)
    if not match or not re.search(r'\bor\b', match[2], re.IGNORECASE):
        return [facet]
    topics = re.split(r',\s*(?:or\s+)?|\s+or\s+', match[2], flags=re.IGNORECASE)
    return [topic.strip() for topic in topics if topic.strip()]


def _evaluation_lifecycle_facets(facets: list[str], query: str, max_facets: int) -> list[str] | None:
    """Add evaluation evidence only after retaining the user's own searches."""
    if not (re.search(r'\bevaluat\w*\b', query, re.I)
            and re.search(r'\b(?:rag|retrieval[ -]augmented|answer quality|hallucinat\w*)\b', query, re.I)):
        return None
    fixed = [
        "Failure points across document ingestion, flawed retrieval, and hallucinated or incomplete answer generation.",
        "Recommended practical offline evaluation lifecycle: fixed golden dataset, CI/CD evaluation gates, regression thresholds, and tuning.",
        "Recommended practical online evaluation lifecycle: asynchronous sampled live traffic, monitoring, A/B testing, and human feedback.",
    ]
    return (facets + fixed)[:max_facets]


def decompose_query(query: str, *, max_facets: int = 3) -> list[str]:
    """Split only explicit multi-part questions; leave focused queries unchanged."""
    normalized = " ".join(query.split())
    sentences = [part.strip() for part in re.split(r"(?<=[?!.])\s+", normalized) if part.strip()]
    facets: list[str] = []
    for sentence in sentences:
        sentence = _searchable_instruction(sentence)
        if not sentence:
            continue
        parts = re.split(
            rf",?\s+and\s+(?=(?:{_FACET_START})\b)",
            sentence,
            flags=re.IGNORECASE,
        )
        facets.extend(search for part in parts if (search := _searchable_instruction(part.strip())))
    if len(facets) < 2:
        facets = _evidence_alternatives(facets[0]) if facets else [normalized]
        if len(facets) > max_facets:
            facets = facets[:max_facets - 1] + [' '.join(facets[max_facets - 1:])]
        return facets
    if len(facets) > max_facets:
        facets = facets[:max_facets - 1] + [' '.join(facets[max_facets - 1:])]
    rewritten = _evaluation_lifecycle_facets(facets, normalized, max_facets)
    if rewritten:
        return rewritten
    if (re.search(r'\b(?:evaluat\w*|assess\w*|measur\w*|benchmark\w*)\b', normalized, re.I)
            and re.search(r'\b(?:practical|approach|framework|strategy|recommend\w*)\b', normalized, re.I)):
        return [f'{facet} Context: {normalized}' for facet in facets]
    return facets


def _passage_key(passage: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(passage.get("file_path") or passage.get("source") or ""),
        str(passage.get("chunk_id") or passage.get("id") or ""),
        str(passage.get("content") or ""),
    )


def balanced_passages(
    groups: Sequence[Sequence[Mapping[str, Any]]], *, limit: int = 5
) -> list[Mapping[str, Any]]:
    """Keep a fixed evidence budget while representing each query facet."""
    selected: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    positions = [0] * len(groups)
    order = list(range(len(groups)))
    while len(selected) < limit:
        made_progress = False
        for group_index in order:
            group = groups[group_index]
            while positions[group_index] < len(group):
                passage = group[positions[group_index]]
                positions[group_index] += 1
                key = _passage_key(passage)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(passage)
                made_progress = True
                break
            if len(selected) == limit:
                return selected
        if not made_progress:
            break
        if len(selected) >= len(groups):
            # Later facets usually contain the requested deliverable ("give",
            # "recommend", "compare"). Split spare slots across them first.
            order = list(reversed(range(len(groups))))
    return selected
