"""Offline retrieval evidence with fixed settings; no model calls or live indexes.

Usage: python scripts/evaluate_lexical_readiness.py SOURCE QUERIES OUTPUT
The CSV uses existing baseline or scaling-experiment column names. Exact answer containment is
a diagnostic, not a semantic relevance grade. Existing output is never replaced.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.core import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'queries', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--chunk-size', type=int, default=700)
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    source, queries, output = (path.resolve() for path in (args.source, args.queries, args.output))
    if output.exists():
        raise FileExistsError(output)
    inputs = {name: {'path': str(path), 'bytes': path.stat().st_size,
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
              for name, path in [('source', source), ('queries', queries)]}
    with queries.open(newline='') as handle:
        questions = list(csv.DictReader(handle))
    for question in questions:
        question['Question'] = question.get('Question', question.get('Q: The Question'))
        question['Expected_Answer'] = question.get('Expected_Answer', question.get('Exact Answer', question.get('Answer')))
        if not question['Question'] or not question['Expected_Answer']:
            raise ValueError('Query CSV requires question and expected-answer fields.')
    config = dict(document_paths=[str(source)], embed_provider='none', backend='lancedb',
                  reranker_type='off', use_contextual=False, chunk_size=args.chunk_size,
                  chunk_overlap=100, top_k=args.top_k, rerank_top_k=args.top_k, force_reindex=True)
    report = dict(inputs=inputs, config=config.copy(), queries=[], api_spend_usd=0,
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    root = Path(__file__).resolve().parents[1]
    code = sorted([*root.glob('rag/**/*.py'), *root.glob('utils/**/*.py'), Path(__file__).resolve()])
    fingerprint = hashlib.sha256()
    for path in code:
        fingerprint.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes() + b'\0')
    report['code_sha256'] = fingerprint.hexdigest()
    report['dependencies'] = {name: version(name) for name in ('lancedb', 'pyarrow', 'pymupdf')}
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
                    retrieved_characters=sum(len(item['original_content']) for item in passages),
                    exact_answer_containment=bool(answer) and any(
                        answer in ' '.join(item['original_content'].casefold().split())
                        for item in passages)))
            except Exception as error:
                report['queries'].append(dict(**question, error=str(error),
                                              seconds=time.perf_counter() - started))
        pipeline.close()
    with output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'queries': len(questions), 'chunks': report['chunks'],
                      'errors': sum('error' in item for item in report['queries']),
                      'exact_answer_containment': sum(item.get('exact_answer_containment', False)
                                                      for item in report['queries']),
                      'output': str(output)}))


if __name__ == '__main__':
    main()
