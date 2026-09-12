"""Offline retrieval evidence with fixed settings; no model calls or live indexes.

Usage: python scripts/evaluate_lexical_readiness.py SOURCE QUERIES OUTPUT
The CSV uses existing baseline or scaling-experiment column names. Exact answer containment is
a diagnostic, not a semantic relevance grade. Existing output is never replaced.
"""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.core import RAGPipeline


def main():
    source, queries, output = map(lambda value: Path(value).resolve(), sys.argv[1:])
    if output.exists():
        raise FileExistsError(output)
    inputs = {name: {'path': str(path), 'bytes': path.stat().st_size,
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
              for name, path in [('source', source), ('queries', queries)]}
    with queries.open(newline='') as handle:
        questions = list(csv.DictReader(handle))
    for question in questions:
        question['Question'] = question.get('Question', question.get('Q: The Question'))
        question['Expected_Answer'] = question.get('Expected_Answer', question.get('Exact Answer'))
        if not question['Question'] or not question['Expected_Answer']:
            raise ValueError('Query CSV requires question and expected-answer fields.')
    config = dict(document_paths=[str(source)], embed_provider='none', backend='lancedb',
                  reranker_type='off', use_contextual=False, chunk_size=700,
                  chunk_overlap=100, top_k=5, rerank_top_k=5, force_reindex=True)
    report = dict(inputs=inputs, config=config.copy(), queries=[], api_spend_usd=0,
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    with tempfile.TemporaryDirectory(prefix='rag-lexical-eval-') as directory:
        config['persist_dir'] = directory
        started = time.perf_counter()
        pipeline = RAGPipeline(config)
        report['initialization_seconds'] = time.perf_counter() - started
        report['chunks'] = pipeline.vector_db.count()
        for question in questions:
            started = time.perf_counter()
            try:
                results = pipeline.search(question['Question'])
                passages = results if isinstance(results, list) else []
                answer = ' '.join(question['Expected_Answer'].casefold().split())
                report['queries'].append(dict(
                    **question, seconds=time.perf_counter() - started, results=results,
                    exact_answer_containment=bool(answer) and any(
                        answer in ' '.join(item['original_content'].casefold().split())
                        for item in passages)))
            except Exception as error:
                report['queries'].append(dict(**question, error=str(error),
                                              seconds=time.perf_counter() - started))
    with output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'queries': len(questions), 'chunks': report['chunks'],
                      'errors': sum('error' in item for item in report['queries']),
                      'exact_answer_containment': sum(item.get('exact_answer_containment', False)
                                                      for item in report['queries']),
                      'output': str(output)}))


if __name__ == '__main__':
    main()
