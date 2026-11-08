# rag-mcp retrieval benchmarks

Offline builders for two Kaggle stages: a GPU retrieval notebook and a hosted
answer/judge task. No benchmark numbers have been produced yet. Dataset schemas,
GPU execution and hosted-model availability still need validation on Kaggle.

## Acceptance contract

| Requirement | Observable check |
| --- | --- |
| Corpus scope | Validate every source row before indexing; qualify questions only when every gold document is loaded; record total/qualified/selected counts |
| Retrieval | Explicit file list into `rag.fetch.get_rag_pipeline_v2`, then `RAGPipeline.search`; audit stored file paths against the full manifest |
| Metrics | Binary document recall@5/@10, nDCG@10 and MRR; duplicates collapse before ranking; hand-calculated test values |
| Durability | Append and fsync each result; retain partial events; skip completed arm/question pairs; report partial metrics and `stopped_early` on failure |
| Hosted answers | At most 30 questions per task; top five saved passages, cited answer, separate model grades correct/partial/wrong with reason |
| Spend | No retries; incremental allocation plus prior spend/reserves; reserve $0.01 for rejected calls of unknown cost; stop on missing successful-call telemetry |
| Reproducibility | Pinned dataset revision and file hashes, README-bound schema, corpus/question hashes, installed rag-mcp commit, model snapshot revisions, flags and counts |
| Scope | Only these scripts/fixtures and `tests/unit/test_benchmarks_*.py`; no legal datasets or private corpora |

A directory lock prevents concurrent writers. All four rag stores are isolated
through `evals.paidrun.isolated_env`; on Kaggle they live under
`/kaggle/working/retrieval_results/stores/<arm>/`. Deleting a generated run folder
rolls back that run. It contains no settings or indexes from the user's machine.

## Offline check

From the checkout root:

```bash
PYTHONPATH=$PWD /home/masih/Desktop/p/rag-mcp/.venv/bin/python \
  -m pytest -q tests/unit/test_benchmarks_*.py
```

The unit suite checks the rag adapter with an offline pipeline double, including
an explicit manifest of 505 files and an omitted-document failure. Hosted calls
and source downloads are replaced at their service boundaries. No GPU model is
loaded by these tests. A separate real BM25 probe was stopped after the installed
LanceDB blocked in `lancedb.connect()`; a standalone connection reproduced the
same stall before indexing. Real LanceDB/GPU execution remains a Kaggle check.

## Stage 1: inspect the dataset on Kaggle

Use a Kaggle account with an available free GPU allocation and internet enabled.
These commands are launch instructions; nothing is uploaded by the builder.
Run them only when ready to send the generated notebook to Kaggle.

```bash
export PYTHONPATH=$PWD
export BENCH_PY=/home/masih/Desktop/p/rag-mcp/.venv/bin/python
export KAGGLE_OWNER=masihmaafi
export BENCH_DIR=/tmp/rag-bench-20261108

"$BENCH_PY" -m evals.benchmarks.build notebook \
  --owner "$KAGGLE_OWNER" --slug rag-bench-enterprise-inspect-20261108 \
  --dataset enterprise --slice github --slice confluence \
  --output "$BENCH_DIR/inspect"
kaggle kernels push --accelerator NvidiaTeslaT4 -p "$BENCH_DIR/inspect"  # Kaggle's T4 machine has two T4s
kaggle kernels status "$KAGGLE_OWNER/rag-bench-enterprise-inspect-20261108"
# Run after status says complete:
kaggle kernels output "$KAGGLE_OWNER/rag-bench-enterprise-inspect-20261108" \
  -p "$BENCH_DIR/inspection"
```

The generated `kernel-metadata.json` is:

```json
{
  "id": "masihmaafi/rag-bench-enterprise-inspect-20261108",
  "title": "rag-bench-enterprise-inspect-20261108",
  "code_file": "benchmark.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": [],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": []
}
```

The install cell executes exactly:

```bash
pip install "ragmcp[full] @ git+https://github.com/MasihMoafi/rag-mcp" \
  huggingface_hub pyarrow
```

`dataset_source/source.json` records the resolved Git commit. `README.md` and
`inspection.json` contain upstream documentation, archive/file hashes, row counts,
field shapes and sample records. Slice arguments are case-insensitive regular
expressions matched against zip paths in that pinned repository tree. If a slice
name does not match, preflight lists available files and stops. Verify that the
patterns selected one or two intended slices, rather than all documents.

Create `$BENCH_DIR/enterprise-schema.json` from those actual fields. **The
following is the synthetic fixture's mapping, not a claimed EnterpriseRAG schema:**

```json
{
  "version": 1,
  "dataset": "enterprise",
  "readme_sha256": "COPY dataset_source/source.json readme_sha256",
  "questions": {
    "file": "questions.jsonl",
    "id": "key",
    "text": "prompt",
    "answer": "answer",
    "gold": "evidence[].key",
    "category": "group"
  },
  "documents": {
    "format": "records",
    "globs": ["extracted/CHOSEN_SLICE/**/*.jsonl"],
    "id": "key",
    "text": ["body"]
  }
}
```

Selectors use dotted object keys; `[]` traverses arrays. `$` selects an entire
record. `rows_key` optionally selects an array inside a JSON wrapper. Text fields
are joined with blank lines. IDs accept strings/integers and compare exactly;
there is no fuzzy matching or inferred basename conversion. Optional `category`
may be omitted. Gold must be an array, empty/null for unlabeled questions. An
absent field, wrong type, duplicate ID, empty text, stale README hash or zero
qualified questions aborts before corpus output or GPU work.

For raw text/Markdown documents, set `documents.format` to `text`, choose globs,
and set `documents.id` to `$stem`, `$name` or `$path` (relative to dataset source).
Zip files extract under `extracted/<zip-stem>/`; paths and symlinks are checked.
Use the selector matching the actual gold IDs. Do not index question or answer
files as documents.

## Stage 1: run the pilot

Read the pinned revision from the inspection download and use it below:

```bash
export DATASET_REVISION=REPLACE_WITH_source_json_revision
"$BENCH_PY" -m evals.benchmarks.build notebook \
  --owner "$KAGGLE_OWNER" --slug rag-bench-enterprise-pilot-20261108 \
  --dataset enterprise --revision "$DATASET_REVISION" \
  --slice github --slice confluence \
  --schema "$BENCH_DIR/enterprise-schema.json" --limit 30 \
  --embedding-size 0.6B --candidates 50 \
  --output "$BENCH_DIR/pilot"
kaggle kernels push --accelerator NvidiaTeslaT4 -p "$BENCH_DIR/pilot"
kaggle kernels status "$KAGGLE_OWNER/rag-bench-enterprise-pilot-20261108"
# Run after status says complete:
kaggle kernels output "$KAGGLE_OWNER/rag-bench-enterprise-pilot-20261108" \
  -p "$BENCH_DIR/pilot-output"
```

The notebook validates the schema, downloads embedding and reranker snapshots,
checks CUDA, then runs `bm25` followed by `hybrid_qwen`. The baseline has no
embedding model and no reranker. Hybrid uses sentence-transformers with
`Qwen/Qwen3-Embedding-0.6B` plus rag-mcp's
`Qwen/Qwen3-Reranker-4B`. `snapshot_download` populates the cache before rag's
`local_files_only=True` reranker load. `--embedding-size 4B` or `8B` scales the
embedding model; those sizes may exceed the free GPU's memory alongside the
reranker. `--limit` scales the question count; `--slice all_documents` selects
the full archive if that is its actual path. Start with the pilot.

Artifacts:

- `bundle/dataset.json`: all dataset provenance and qualification counts.
- `retrieval_results/manifest.json`: run identity, models, flags and counts.
- `retrieval_results/retrieval.jsonl`: ranked document IDs, aligned scores,
  first five passages with locations, question/gold answer, latency and errors.
- `retrieval_results/metrics.json`: completion/error counts per arm and macro
  metrics for completed questions. Failed/incomplete questions are not silently
  scored as successes; compare arms on the same completed question IDs.

The library retrieves/reranks up to `--candidates` passages. Document metrics
collapse multiple passages from the same document. MRR is truncated by this
candidate window, which is recorded in the manifest; recall@10 can be limited
by duplicate chunks. BM25 scores in output are rag's RRF scores; hybrid scores
are the reranker's scores. A missing reranker ranking is an arm failure.
Indexing latency is outside per-question search latency. No hosted answer
model is called in this stage.

For retrieval resume, build a new notebook/slug/output folder using the same
schema, pinned dataset revision, slices and flags, adding:

```bash
--resume-from "$BENCH_DIR/pilot-output/retrieval_results/retrieval.jsonl"
```

The full ledger is embedded byte for byte. Stores rebuild in the new Kaggle run;
completed questions are skipped. A different installed rag-mcp commit or model
revision fails identity preflight rather than mixing results. New runs install
GitHub main, so archive the original environment to resume if main has moved.

## FreshStack and FRAMES

`datasets.prepare` also accepts `--dataset freshstack` and `--dataset frames`.
For FreshStack, it searches Hugging Face for `FreshStack` at runtime. Multiple
matches stop with a candidate list; select `--hf-id` after verifying the dataset
card against [the FreshStack paper](https://arxiv.org/abs/2504.13128). No guessed
Hugging Face ID is hardcoded. FRAMES defaults to `google/frames-benchmark`.
An inspection build without `--file` downloads the README and saves the full
repository inventory in `dataset_source/source.json`. Pass explicit `--file`
paths from that inventory in the next build; JSON, JSONL
and Parquet are supported, including separate corpus and question files.
Use another README-bound mapping with `dataset` set accordingly. The notebook
builder forwards `--hf-id` and repeated `--file` options to the same loader.
EnterpriseRAG can also use its confirmed Hugging Face mirror through `--hf-id`
with explicit `--file` paths if the GitHub repository links to externally hosted
documents instead of containing the zips itself.

For FRAMES, map `questions.gold` to the actual array of Wikipedia URLs and omit
`documents`. The loader validates English Wikipedia URLs, fetches article text
through `en.wikipedia.org/w/api.php`, follows redirects and stores full API
responses under `dataset_source/wikipedia-cache/`. Only selected questions'
articles are fetched. Their text hashes and cached response hashes are recorded;
Wikipedia text is a current snapshot, not the benchmark's historical snapshot.
Every FRAMES retrieval row, answer row and run manifest is labeled **may be
memorised**. Treat it as a comparison rather than contamination-resistant proof.

## Stage 2: build, run and download a hosted answer task

Choose two model IDs available to the Kaggle Benchmarks account. Check the
remaining free credit and current provider prices before choosing `--ceiling`
and `--call-budget`. No personal provider key is used. This implementation does
not promise free quota is available; do not run beyond the account's free credit.
`--call-budget` is a conservative per-call reservation, **not a provider-side
hard price cap**. Actual cost can exceed an estimate; that stops further calls.

```bash
export ANSWER_MODEL=REPLACE_WITH_AVAILABLE_KAGGLE_MODEL_ID
export JUDGE_MODEL=REPLACE_WITH_DISTINCT_KAGGLE_MODEL_ID
export ANSWER_TASK=rag-bench-enterprise-hybrid-20261108

"$BENCH_PY" -m evals.benchmarks.build answers \
  --retrieval "$BENCH_DIR/pilot-output/retrieval_results/retrieval.jsonl" \
  --retrieval-manifest "$BENCH_DIR/pilot-output/retrieval_results/manifest.json" \
  --arm hybrid_qwen --model "$ANSWER_MODEL" --judge "$JUDGE_MODEL" \
  --task "$ANSWER_TASK" --limit 30 --offset 0 \
  --ceiling 0.50 --call-budget 0.05 \
  --output "$BENCH_DIR/answer-task.py"
kaggle benchmarks tasks push "$ANSWER_TASK" -f "$BENCH_DIR/answer-task.py" \
  --wait 540 --poll-interval 30
kaggle benchmarks tasks status "$ANSWER_TASK"
kaggle benchmarks tasks download "$ANSWER_TASK" -o "$BENCH_DIR/answers-output"
```

`tasks push` executes the source, including its `.run(kbench.llms[MODEL])` call.
It can start the first run automatically. Do not immediately issue a duplicate
`tasks run`. If status confirms no run was started, the explicit command is:

```bash
kaggle benchmarks tasks run "$ANSWER_TASK" -m "$ANSWER_MODEL" \
  --wait 540 --poll-interval 30
```

Run each arm as its own task. Later shards use `--offset 30`, `60`, etc. with
`--limit 30`. The task uses fresh chats for answering and grading so the answer
model never sees the gold answer and the judge does not inherit its conversation.
Citation IDs must refer to the supplied top-five passages. Grading is model-based,
not a human correctness guarantee. Grade counts retain reasons in the ledger.

Download `answer_results/answers.jsonl`, `manifest.json` and `metrics.json` from
the returned bundle. For resume, use that **latest full, newline-terminated**
JSONL as `--resume-from`, with the same arm/shard/models and a new task name/output
file. `--ceiling` is the additional allocation. All prior calls, failures, costs
and $0.01 unknown rejected-call reserves remain in the parent prefix. A saved
answer is reused if grading failed; completed questions incur no new calls.
A successful/interrupted call without known cost stops further calls until its
cost is reconciled. Keep the original ledger when reconciling; never drop paid
partial events. Divergent parent prefixes and torn JSONL are rejected.

After work starts, failures, Ctrl-C and SIGTERM are caught; the runner records
`stopped_early`, writes metrics for finished questions and returns normally.
Model identity, schema, GPU/download, manifest and ledger validation are preflight
and may raise. SIGKILL, kernel eviction and unavailable disk cannot be made
durable by Python; fsynced evidence is retained only if Kaggle preserves the
filesystem. Benchmarks discards output on task ERROR, so inspect partial outputs
and resume instead of raising to signal an incomplete result.

## Remaining external validation

Confirm actual field maps, source zip locations and the FreshStack repository on
Kaggle; none was accessible in the offline implementation environment. Confirm
current GPU quota, both hosted model IDs, response/cost telemetry and available
free credit before launching. Public claims need the saved manifests and raw
JSONL, plus the selected question set and any incomplete-arm exclusions. The
builders create private Kaggle artifacts; publishing results is a separate step.
