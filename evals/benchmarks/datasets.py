"""Runtime source inspection and strict, explicit dataset field contracts.

No upstream field names are assumed. Inspect on Kaggle, bind the actual schema,
then normalize all rows before allowing a retrieval run.
"""

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

from .common import digest, file_hash, write_json

USER_AGENT = "rag-mcp-bench/1.0 (https://github.com/MasihMoafi/rag-mcp)"


def get_json(url):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response, destination.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)


def safe_path(root, relative):
    target = (Path(root) / relative).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path escapes dataset directory: {relative}")
    return target


def records(path, rows_key=None):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if rows_key:
            value = select(value, rows_key)
        return value if isinstance(value, list) else [value]
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    raise ValueError(f"Unsupported record file: {path}")


def select(row, selector):
    """Dotted keys, with [] to traverse an array (e.g. citations[].document.id)."""
    if selector == "$":
        return row
    head, dot, tail = selector.partition(".")
    array = head.endswith("[]")
    key = head[:-2] if array else head
    if not isinstance(row, dict) or key not in row:
        raise ValueError(
            f"Schema: missing {selector!r}; observed keys: "
            f"{sorted(row) if isinstance(row, dict) else type(row).__name__}"
        )
    value = row[key]
    if array:
        if not isinstance(value, list):
            raise ValueError(f"Schema: {head} must be an array")
        values = [select(item, tail) if dot else item for item in value]
        return [
            nested
            for item in values
            for nested in (item if isinstance(item, list) else [item])
        ]
    return select(value, tail) if dot else value


def shape(value):
    if isinstance(value, dict):
        return {key: shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return type(value).__name__


def inspect_source(root):
    root = Path(root)
    inventory = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"inspection.json", "source.json"}:
            continue
        row = {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        }
        if path.suffix in {".json", ".jsonl", ".parquet"}:
            data = records(path)
            row.update(
                rows=len(data),
                sample_shape=shape(data[0]) if data else None,
                sample=data[:1],
            )
        inventory.append(row)
    write_json(root / "inspection.json", inventory)
    return inventory


def prepare(dataset, root, slices=(), revision="main", hf_id=None, files=()):
    """Download only selected sources, pin revision, preserve README + hashes."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    if dataset == "enterprise" and hf_id is None:
        # The repository tree is too large for one GitHub listing (truncated); the release assets
        # hold the questions and per-source slices. 'main' means the latest release.
        repo = "onyx-dot-app/EnterpriseRAG-Bench"
        api = f"https://api.github.com/repos/{repo}"
        release = get_json(
            f"{api}/releases/latest"
            if revision == "main"
            else f"{api}/releases/tags/{quote(revision, safe='')}"
        )
        tag = release["tag_name"]
        commit = get_json(f"{api}/commits/{quote(tag, safe='')}")["sha"]
        assets = {
            asset["name"]: asset["browser_download_url"] for asset in release["assets"]
        }
        download(
            f"https://raw.githubusercontent.com/{repo}/{commit}/README.md",
            root / "README.md",
        )
        if "questions.jsonl" not in assets or not slices:
            raise ValueError(
                f"Expected questions.jsonl and explicit --slice; assets={sorted(assets)}"
            )
        selected = []
        for pattern in slices:
            matches = sorted(
                n
                for n in assets
                if n.endswith(".zip") and re.search(pattern, n, re.IGNORECASE)
            )
            if not matches:
                raise ValueError(
                    f"No slice zip matches {pattern!r}; assets={sorted(assets)}"
                )
            selected.extend(matches)
        for name in dict.fromkeys(["questions.jsonl", *selected]):
            download(assets[name], root / name)
        source = {
            "dataset": dataset,
            "repository": repo,
            "release": tag,
            "revision": commit,
            "slices": sorted(set(selected)),
        }
    else:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        if dataset == "freshstack" and hf_id is None:
            candidates = [item.id for item in api.list_datasets(search="FreshStack")]
            if len(candidates) != 1:
                raise ValueError(
                    f"FreshStack dataset id needs selection from runtime results: {candidates}; "
                    "rerun with --hf-id (check against arXiv:2504.13128)"
                )
            hf_id = candidates[0]
        hf_id = hf_id or "google/frames-benchmark"
        commit = api.dataset_info(hf_id, revision=revision).sha
        names = api.list_repo_files(hf_id, repo_type="dataset", revision=commit)
        selected = list(files)
        for pattern in slices:
            matches = [
                n
                for n in names
                if n.endswith(".zip") and re.search(pattern, n, re.IGNORECASE)
            ]
            if not matches:
                raise ValueError(f"No HF slice zip matches {pattern!r}; files={names}")
            selected.extend(matches)
        for name in dict.fromkeys(["README.md", *selected]):
            if name not in names:
                raise ValueError(f"Missing dataset file {name!r}; files={names}")
            cached = Path(
                hf_hub_download(hf_id, name, repo_type="dataset", revision=commit)
            )
            target = safe_path(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(cached.read_bytes())
        source = {
            "dataset": dataset,
            "repository": hf_id,
            "revision": commit,
            "files": selected,
            "available_files": names,
        }
    # Extract each zip separately: filenames can overlap across source slices.
    archives = sorted(root.rglob("*.zip"))
    if len({path.stem for path in archives}) != len(archives):
        raise ValueError(
            "Slice archives have duplicate basenames; select unambiguous slices"
        )
    for path in archives:
        target = root / "extracted" / path.stem
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                safe_path(target, member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive contains a symbolic link")
            archive.extractall(target)
    source["readme_sha256"] = file_hash(root / "README.md")
    write_json(root / "source.json", source)
    inspect_source(root)
    return source


def nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Schema: {label} must be a nonempty string")
    return value


def doc_id(value):
    if type(value) not in (str, int) or not str(value).strip():
        raise ValueError(
            "Schema: document/question id must be a nonempty string or integer"
        )
    return str(value)


def wiki_title(url):
    parsed = urlparse(url)
    if parsed.hostname != "en.wikipedia.org" or not parsed.path.startswith("/wiki/"):
        raise ValueError(f"Schema: expected English Wikipedia article URL: {url!r}")
    return unquote(parsed.path[len("/wiki/") :]).replace("_", " ")


def wikipedia_text(url, cache):
    """Cache both API response and resolved title; URL stays the gold document id."""
    title = wiki_title(url)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (digest(url) + ".json")
    if not path.exists():
        data = get_json(
            "https://en.wikipedia.org/w/api.php?action=query&format=json&"
            "prop=extracts&explaintext=1&redirects=1&titles=" + quote(title)
        )
        write_json(path, data)
    data = json.loads(path.read_text())
    pages = list(data["query"]["pages"].values())
    if len(pages) != 1 or "missing" in pages[0]:
        raise ValueError(f"Wikipedia article unavailable: {url}")
    return nonempty(pages[0].get("extract"), f"Wikipedia text for {url}")


def normalize(root, schema_path, output, limit=30):
    """Validate ALL rows and gold references before creating any corpus output."""
    root, output = Path(root), Path(output)
    source = json.loads((root / "source.json").read_text())
    schema = json.loads(Path(schema_path).read_text())
    if schema.get("version") != 1 or schema.get("dataset") != source["dataset"]:
        raise ValueError("Schema version/dataset differs from source")
    if schema.get("readme_sha256") != file_hash(root / "README.md"):
        raise ValueError("Schema must bind the inspected upstream README sha256")
    if limit < 1:
        raise ValueError("--limit must be positive")
    qs = schema["questions"]
    raw_questions = records(safe_path(root, qs["file"]), qs.get("rows_key"))
    questions = []
    for row in raw_questions:
        gold = select(row, qs["gold"])
        if gold is None:
            gold = []
        if not isinstance(gold, list):
            raise ValueError(
                "Schema: gold selector must return an array (or null for no labels)"
            )
        question = {
            "id": doc_id(select(row, qs["id"])),
            "question": nonempty(select(row, qs["text"]), "question text"),
            "gold_answer": nonempty(select(row, qs["answer"]), "gold answer"),
            "gold_doc_ids": list(dict.fromkeys(doc_id(g) for g in gold)),
            "category": select(row, qs["category"]) if qs.get("category") else None,
        }
        questions.append(question)
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Schema: empty questions or duplicate question ids")
    documents = {}
    if source["dataset"] == "frames":
        # Validate URLs for the entire schema first; fetch just the selected pilot.
        for question in questions:
            for url in question["gold_doc_ids"]:
                wiki_title(url)
        for question in [q for q in questions if q["gold_doc_ids"]][:limit]:
            for url in question["gold_doc_ids"]:
                if url not in documents:
                    documents[url] = wikipedia_text(url, root / "wikipedia-cache")
    else:
        ds = schema["documents"]
        paths = sorted(
            {p for glob in ds["globs"] for p in root.glob(glob) if p.is_file()}
        )
        if not paths:
            raise ValueError("Schema: document globs match no files")
        for path in paths:
            if ds["format"] == "text":
                selectors = {
                    "$stem": path.stem,
                    "$name": path.name,
                    "$path": str(path.relative_to(root)),
                }
                if ds["id"] not in selectors:
                    raise ValueError(
                        "Schema: text document id must be $stem, $name or $path"
                    )
                rows = [(selectors[ds["id"]], path.read_text(encoding="utf-8"))]
            elif ds["format"] == "records":
                rows = [
                    (
                        select(row, ds["id"]),
                        "\n\n".join(
                            nonempty(select(row, field), field) for field in ds["text"]
                        ),
                    )
                    for row in records(path, ds.get("rows_key"))
                ]
            else:
                raise ValueError("Schema: document format must be text or records")
            for key, text in rows:
                key = doc_id(key)
                if key in documents:
                    raise ValueError(f"Schema: duplicate document id {key!r}")
                documents[key] = nonempty(text, f"document {key}")
    qualified = [
        q
        for q in questions
        if q["gold_doc_ids"] and set(q["gold_doc_ids"]).issubset(documents)
    ]
    if not qualified:
        raise ValueError(
            "Schema/slices: zero questions have all gold documents in corpus; "
            "check exact document-id mapping before indexing"
        )
    selected = qualified[:limit]
    output.mkdir(parents=True, exist_ok=False)
    corpus = output / "corpus"
    corpus.mkdir()
    mapping = {}
    for key, text in documents.items():
        name = digest(key) + ".txt"
        (corpus / name).write_text(text, encoding="utf-8")
        mapping[name] = key
    write_json(output / "documents.json", mapping)
    with (output / "questions.jsonl").open("w") as stream:
        for q in selected:
            stream.write(json.dumps(q, ensure_ascii=False) + "\n")
    manifest = dict(
        source,
        schema_sha256=file_hash(schema_path),
        source_files=inspect_source(root),
        documents=len(documents),
        questions_total=len(questions),
        questions_without_gold=sum(not q["gold_doc_ids"] for q in questions),
        questions_qualified=len(qualified),
        questions_selected=len(selected),
        qualified_categories=dict(Counter(str(q["category"]) for q in qualified)),
        corpus_sha256=digest(documents),
        questions_sha256=digest(selected),
        caveat="may be memorised" if source["dataset"] == "frames" else None,
    )
    write_json(output / "dataset.json", manifest)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "documents",
                    "questions_total",
                    "questions_qualified",
                    "questions_selected",
                )
            }
        )
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument(
        "--dataset", choices=("enterprise", "freshstack", "frames"), required=True
    )
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--slice", action="append", default=[])
    prepare_parser.add_argument("--revision", default="main")
    prepare_parser.add_argument("--hf-id")
    prepare_parser.add_argument("--file", action="append", default=[])
    normalize_parser = sub.add_parser("normalize")
    normalize_parser.add_argument("--root", type=Path, required=True)
    normalize_parser.add_argument("--schema", type=Path, required=True)
    normalize_parser.add_argument("--output", type=Path, required=True)
    normalize_parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            args.dataset, args.root, args.slice, args.revision, args.hf_id, args.file
        )
    else:
        normalize(args.root, args.schema, args.output, args.limit)


if __name__ == "__main__":
    main()
