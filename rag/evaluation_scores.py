"""Retrieval scores for one evaluation batch against passages a person marked relevant.

Right answers are the relevant labels from every batch over the same sources, so a batch
run with another configuration is scored against the same answers.
"""
from __future__ import annotations

import math

from rag.evaluation_labels import EvaluationLabelStore

_METRICS = ('hit', 'recall', 'mrr', 'ndcg', 'precision')


def _matches(passage, span):
    """Same source and at least half of the shorter span overlaps (chunk edges move between configurations)."""
    if passage['source_id'] != span['source_id']:
        return False
    overlap = min(passage['end_offset'], span['end_offset']) - max(passage['start_offset'], span['start_offset'])
    shorter = min(passage['end_offset'] - passage['start_offset'], span['end_offset'] - span['start_offset'])
    return overlap > 0 and overlap * 2 >= shorter


def _judgments(workspace_id, revision, labels):
    """Right answers per question (groups of spans that match each other) and every judged span."""
    directory = labels.batches.path(workspace_id, '0' * 32).parent
    answers, judged = {}, {}
    for path in sorted(directory.glob('*.labels.json')):
        batch_id = path.name.removesuffix('.labels.json')
        try:
            if labels.batches.get(workspace_id, batch_id)['workspace_revision'] != revision:
                continue
            items = labels.get(workspace_id, batch_id)['labels']
        except (OSError, ValueError):
            continue
        for item in items:
            judged.setdefault(item['question_id'], []).append(item)
            if item['label'] != 'relevant':
                continue
            groups = answers.setdefault(item['question_id'], [])
            group = next((group for group in groups if any(_matches(item, span) for span in group)), None)
            if group is None:
                groups.append([item])
            else:
                group.append(item)
    return answers, judged


def _score(relevant_ranks, answers_found, answer_count, passage_count):
    if not passage_count:
        return {'hit': 0.0, 'recall': 0.0, 'mrr': 0.0, 'ndcg': 0.0, 'precision': 0.0}
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(answer_count, passage_count) + 1))
    return {
        'hit': 1.0 if relevant_ranks else 0.0,
        'recall': answers_found / answer_count,
        'mrr': 1 / relevant_ranks[0] if relevant_ranks else 0.0,
        'ndcg': sum(1 / math.log2(rank + 1) for rank in relevant_ranks) / ideal,
        'precision': len(relevant_ranks) / passage_count,
    }


def score_batch(workspace_id: str, batch_id: str, labels: EvaluationLabelStore | None = None) -> dict:
    labels = labels or EvaluationLabelStore()
    batch = labels.batches.get(workspace_id, batch_id)
    answers, judged = _judgments(workspace_id, batch['workspace_revision'], labels)
    questions, unjudged_passages, missing = [], 0, 0
    for row in batch['rows']:
        groups = answers.get(row['question_id'], [])
        entry = {'question_id': row['question_id'], 'run_id': row['run_id'], 'gold_count': len(groups),
                 'passage_count': row['passage_count'], 'relevant_ranks': [], 'evidence_available': True,
                 **dict.fromkeys(_METRICS)}
        questions.append(entry)
        if row['status'] != 'success':
            continue
        result = labels.results.read(row['run_id']) if row['passage_count'] else {'evidence': []}
        if not result or result.get('workspace_id', workspace_id) != workspace_id or not isinstance(result.get('evidence'), list):
            entry['evidence_available'] = False
            missing += 1
            continue
        passages = sorted((item for item in result['evidence'] if isinstance(item, dict)), key=lambda item: item['rank'])
        found = set()
        for passage in passages:
            hits = [index for index, group in enumerate(groups) if any(_matches(passage, span) for span in group)]
            if hits:
                entry['relevant_ranks'].append(passage['rank'])
                found.update(hits)
            elif not any(_matches(passage, span) for span in judged.get(row['question_id'], [])):
                unjudged_passages += 1
        if groups:
            entry.update(_score(entry['relevant_ranks'], len(found), len(groups), len(passages)))
    scored = [entry for entry in questions if entry['hit'] is not None]
    mean = (lambda key: round(sum(entry[key] for entry in scored) / len(scored), 4)) if scored else (lambda key: None)
    return {
        'schema_version': 1, 'workspace_id': workspace_id, 'batch_id': batch_id,
        'basis': 'human_relevance_labels',
        'summary': {'judged_questions': len(scored),
                    'unjudged_questions': len(questions) - len(scored) - missing,
                    'hit_rate': mean('hit'), 'recall': mean('recall'), 'mrr': mean('mrr'),
                    'ndcg': mean('ndcg'), 'precision': mean('precision'),
                    'unjudged_passages': unjudged_passages, 'missing_evidence': missing},
        'questions': questions,
    }
