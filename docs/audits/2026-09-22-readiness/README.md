# Production-readiness baseline — 2026-09-22

The [specification](../../specs-2026-09-22-production-readiness.md) defines the
held-out evaluation and private observability contracts. Neither is implemented
or passed by this checkpoint. No runtime files, provider configuration, indexes,
or existing dirty files were changed. No paid requests were made.

## Fresh result

**75 passed, 6 failed, 16 warnings, 11.19 seconds** on the existing selected
contracts. This is not the full suite, answer-quality evaluation, or a latency
benchmark. Keep the red baseline as evidence; do not reinterpret it as green.

Command from the repository root:

```sh
.venv/bin/python docs/audits/2026-09-22-readiness/run_baseline.py
```

The runner uses the existing nine test files, temporary configuration/indexes,
fake providers and a Python socket guard allowing loopback only (for companion
HTTP checks). Child Python processes inherit the guard. It is a test safeguard,
not an OS-level network sandbox. No new test cases were added.

| Failures | Observation | Required next action |
| --- | --- | --- |
| 2 in `test_retrieval_log_privacy.py` | Expected `top_k=20`, actual 50 | Align the default assertion with the 50-candidate contract, retaining secret-marker checks |
| 4 in `test_incremental_index.py` | Fixture expects a public passage list; call reaches synthesis and fails on missing credentials | Verify indexing through the internal retrieval boundary and separately stub/check the public answer boundary; never supply real credentials to these tests |

The four indexing failures occur before assertions can verify refresh behavior.
They do not establish either successful or broken incremental updates on the
current answer-only path. The two privacy failures occur before the final log
checks; their names alone are not evidence of a privacy leak.

## Files and provenance

- `baseline-manifest.json`: baseline HEAD, status, staged/tracked diff hashes and
  hashes of every pre-existing modified/untracked file. It captures 167 files
  totaling 586,868,646 bytes. Git status groups some untracked directories.
- `environment.json`: actual installed versions and lockfile identity, without
  credentials or user configuration.
- `run_baseline.py`, `pytest-run.json`, `pytest.log`: procedure, command/exit code,
  elapsed time, log hash, and complete output. A rerun replaces the two output
  files; the local commit preserves this original observation.
- `historical-rerank-summary.json`: recomputed metrics and per-request usage from
  the existing five-case synthetic Jev/Cohere experiment. Both achieved 5/5
  Hit@1 and Hit@5. These are historical observations, not a new experiment.
  Raw evidence remains at the recorded `/tmp` path with SHA-256; this summary is
  not a durable replacement for that raw file and must not justify promotion.
- `preservation-check.json`: final hash comparison of all pre-existing dirty
  files against the initial manifest.

Owner requirements were recovered from local `.elpis/history.jsonl` and the
previous session rollout, whose IDs are recorded in the spec. Full conversation,
private configurations, account identifiers and keys were not copied here.
The preceding session's failed deployment-chapter transfer is transcript evidence;
it lacks a preserved stage-level trace here, so its first evidence-loss stage
remains unknown. Do not promote an architectural hypothesis to a finding.

## Decision

Do not implement Jev or broaden refactoring from these results. The next narrow
checkpoint is repairing stale test boundaries and adding local hosted-pipeline
traces against the specified contracts. Build/review/freeze unseen evaluation
data before any quality claim or paid comparison. Existing historical retrieval
scores and companion dashboard counters cannot establish hosted answer readiness.
New spend is $0; no push, deployment, download or remote export occurred.
