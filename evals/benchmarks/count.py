"""Exact size of a zipped corpus as rag would embed it: documents, characters, chunks and tokens.

Uses rag's own chunker and cleaning, and makes no network or model calls.

    python -m evals.benchmarks.count ZIP --tokenizer tokenizer.json [--only github confluence]
"""

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer


def searchable_chunks(text, name="document.txt"):
    """The strings rag sends to the embedding model for one .txt document."""
    from rag.chunking import chunk_document_spans
    from rag.ingestion import _PROSE_EXTENSIONS, clean_text

    extension = Path(name).suffix.lower()
    chunks = []
    for item in chunk_document_spans(
        text, file_path=name, chunk_size=700, chunk_overlap=100
    ):
        if not item["content"].strip():
            continue
        clean = item["content"].replace("\f", " ")
        searchable = (
            clean_text(clean, markdown=False)
            if extension in _PROSE_EXTENSIONS
            else clean
        )
        if searchable:
            chunks.append(searchable)
    return chunks


def count(archive_path, tokenizer, only=(), batch_size=2000):
    totals = defaultdict(
        lambda: {"documents": 0, "characters": 0, "chunks": 0, "tokens": 0}
    )
    pending, owners = [], []

    def flush():
        for owner, encoding in zip(
            owners,
            tokenizer.encode_batch(pending, add_special_tokens=False),
            strict=True,
        ):
            totals[owner]["tokens"] += len(encoding.ids)
        pending.clear()
        owners.clear()

    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            source = info.filename.split("/", 1)[0]
            if (
                info.is_dir()
                or not info.filename.endswith(".txt")
                or (only and source not in only)
            ):
                continue
            text = archive.read(info).decode("utf-8")
            chunks = searchable_chunks(text)
            row = totals[source]
            row["documents"] += 1
            row["characters"] += len(text)
            row["chunks"] += len(chunks)
            pending.extend(chunks)
            owners.extend([source] * len(chunks))
            if len(pending) >= batch_size:
                flush()
    if pending:
        flush()
    report = {"sources": dict(sorted(totals.items()))}
    report["total"] = {
        key: sum(row[key] for row in totals.values())
        for key in ("documents", "characters", "chunks", "tokens")
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=())
    args = parser.parse_args()
    print(
        json.dumps(
            count(args.archive, Tokenizer.from_file(str(args.tokenizer)), args.only),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
