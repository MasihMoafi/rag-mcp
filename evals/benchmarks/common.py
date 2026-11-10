"""Document-level metrics and append-only evidence shared by both stages."""

import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal


# Masih's rag settings: 20 passages for the answer, one neighbouring chunk on each side.
ANSWER_PASSAGES = 20
CONTEXT_RADIUS = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    """Replace a summary atomically; the ledger is the authoritative evidence."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def read_ledger(path):
    raw = Path(path).read_bytes() if Path(path).exists() else b''
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Ledger must end in a newline; retain the torn file for manual recovery')
    return raw, [json.loads(line) for line in raw.splitlines()]


class Ledger:
    def __init__(self, path, identity, seed=b''):
        self.path, self.identity = Path(path), identity
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if seed and not self.path.exists():
            with self.path.open('xb') as stream:
                stream.write(seed)
        raw, self.events = read_ledger(self.path)
        if not raw.startswith(seed):
            raise ValueError('Existing ledger diverges from the full resume parent')
        if any(event.get('identity') != identity for event in self.events):
            raise ValueError('Resume identity differs: dataset, models, flags or questions changed')
        self.parent_hash = hashlib.sha256(raw).hexdigest() if raw else None

    def append(self, event):
        event = dict(event, identity=self.identity)
        line = json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)

    def completed(self):
        return {(e['arm'], e['question_id']) for e in self.events if e.get('event') == 'complete'}


@contextlib.contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.benchmark.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another benchmark owns this output directory') from None
        yield


@contextlib.contextmanager
def graceful_signals():
    def stop(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def retrieval_metrics(ranked, gold):
    """Binary document relevance; collapse duplicate chunks before ranking."""
    ranked = list(dict.fromkeys(ranked))
    gold = set(gold)
    if not gold:
        raise ValueError('Retrieval metrics require nonempty gold documents')
    hits = [int(doc in gold) for doc in ranked]
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits[:10]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
    return {'recall@5': sum(hits[:5]) / len(gold),
            'recall@10': sum(hits[:10]) / len(gold), 'nDCG@10': dcg / ideal,
            'MRR': next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)}


def retrieval_summary(events, questions, arms, stopped):
    gold = {q['id']: q['gold_doc_ids'] for q in questions}
    report = {'stopped_early': stopped, 'planned_per_arm': len(questions), 'arms': {}}
    for arm in arms:
        done = {e['question_id']: e for e in events
                if e.get('event') == 'complete' and e['arm'] == arm}
        scores = [retrieval_metrics(e['ranked_doc_ids'], gold[qid]) for qid, e in done.items() if gold[qid]]
        report['arms'][arm] = {
            'completed': len(done), 'remaining': len(questions) - len(done),
            'error_attempts': sum(e.get('event') == 'error' and e.get('arm') == arm for e in events),
            'macro_completed_only': {key: sum(s[key] for s in scores) / len(scores) if scores else None
                                     for key in ('recall@5', 'recall@10', 'nDCG@10', 'MRR')}}
    return report


def finish_safely(output, ledger, summary):
    """Keep post-start reporting failures from turning a hosted task into ERROR."""
    try:
        write_json(Path(output) / 'metrics.json', summary)
    except BaseException as error:
        print('STOPPED_EARLY summary write failed:', repr(error), flush=True)
    try:
        ledger.append({'event': 'stopped_early' if summary.get('stopped_early') else 'run_finished',
                       'reason': summary.get('stopped_early')})
    except BaseException as error:
        print('STOPPED_EARLY ledger write failed:', repr(error), flush=True)
    return summary
