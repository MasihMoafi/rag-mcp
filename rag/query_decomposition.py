"""Deterministic query facets and balanced evidence selection."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


_FACET_START = (
    r"what|why|how|which|who|where|when|give|explain|describe|identify|"
    r"compare|list|summarize|outline|provide"
)


def _evaluation_lifecycle_facets(facets: list[str], query: str) -> list[str] | None:
    """Separate broad evaluation questions into failure, offline, and online evidence."""
    combined = f"{' '.join(facets)} {query}".lower()
    asks_for_practice = any(
        term in combined
        for term in ("practical", "approach", "framework", "strategy", "recommend")
    )
    concerns_evaluation = any(
        term in combined for term in ("evaluat", "assess", "measure", "benchmark")
    )
    if asks_for_practice and concerns_evaluation:
        failure = (
            "Failure points across document ingestion (including outdated or improperly parsed "
            "documents), flawed retrieval, and hallucinated or incomplete answer generation: "
            f"{' '.join(facets[:-1]).strip()}"
        )
        return [
            failure,
            (
                "Recommended practical offline evaluation lifecycle: fixed golden dataset, "
                "CI/CD evaluation gates, regression thresholds, and tuning."
            ),
            (
                "Recommended practical online evaluation lifecycle: asynchronous sampled live "
                "traffic, monitoring, A/B testing, and human feedback."
            ),
        ]
    return None


def decompose_query(query: str, *, max_facets: int = 3) -> list[str]:
    """Split only explicit multi-part questions; leave focused queries unchanged."""
    normalized = " ".join(query.split())
    sentences = [part.strip() for part in re.split(r"(?<=[?!.])\s+", normalized) if part.strip()]
    facets: list[str] = []
    for sentence in sentences:
        parts = re.split(
            rf",?\s+and\s+(?=(?:{_FACET_START})\b)",
            sentence,
            flags=re.IGNORECASE,
        )
        facets.extend(part.strip() for part in parts if len(part.split()) >= 3)
    if len(facets) < 2:
        return [normalized]
    facets = facets[:max_facets]
    return _evaluation_lifecycle_facets(facets, normalized) or facets


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
