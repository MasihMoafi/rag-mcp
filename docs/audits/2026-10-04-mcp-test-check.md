# Check of the rag MCP test (2026-10-04)

Evidence: the agent transcript, the rag traces in `~/.local/state/ragmcp/traces`, the client log, the source, and 4 repeat queries.

## Claims that are correct

1. The 500-document limit is in the source: `rag/workspaces.py:93` (no override) and `rag/fetch.py:260` for folders.
2. All 413 documents were Rust. They are exactly `codex-rs/core/src` (413 `.rs` files). No TUI code.
3. `answer_max_tokens` is 700 (`rag/settings.py:125`). `set_rag_settings` cannot set it or the timeouts.
4. `get_workspaces` has no `limit` or `offset`. One call gave 69,324 characters.
5. A refusal gives no reason. The prompt orders one exact sentence (`rag/openrouter_pipeline.py:276`).
6. The `top` change is fully reverted. Only the revision hash is different in the stored configuration.
7. My repeat queries agree with its final results. Ledger keys: correct, 3.6 s. Kubernetes: refusal, 2.8 s. Context build: vague, 3.0 s. A precise code question: correct, 3.1 s.

## Claims that are wrong or too strong

1. **"Missing docs" is the cause.** This is true only for the Ledger question. The Ledger keys are in the TUI crate and the docs. The context question is about core code, and the workspace has that code.
2. **"Synthesis refuses when retrieval finds the right passages."** The two tools got the same 5 passages. They came from the correct files, but no passage states the answer: a test, an import list, two comments. The prompt does not accept topical similarity as evidence, so the refusal is correct. The weak part is retrieval, not synthesis.
3. **"Its mistake was to query the whole workspace."** Too strong. A workspace query has no `source_ids` field, so an agent cannot make it smaller. The step it did not do: find what the workspace holds before it asks a TUI question.
4. **"Two queries started together and took over 120 s."** Wrong. Claude Code sent the calls one at a time. 120 s is the Claude Code limit before it moves a call to the background. Server times: 257 s, 4.2 s, 3.5 s. The second and third calls waited behind the first.
5. **Cause of the vague answer not found.** It reported the vague code answer. It did not find the cause: after the docs went in, all passages for that question came from `GUIDE.md` and `context.md`. The docs now push code out of broad questions.

## Items it did not check (I checked them)

1. **Cause of the 257 s query.** `index.build` took 254 s. Each workspace edit copies all documents into a new folder (`rag/workspaces.py:131`), and a query reads from that folder (`rag/execution.py:180`). The index key changes, so the incremental refresh cannot apply. The build embedded all 12,748 chunks again to add 4 files.
2. **The trace is incomplete.** It shows 1 embedding batch (512 chunks, 6.4 s, $0.0012) of about 25. Thus approximately 247 s and 90% of the cost are not in the trace. The cause is not checked.
3. **Old data stays on disk.** The old workspace index (165 MB) and 4 old document copies stay. The workspace folder is now 38 MB.
4. **The live server runs code that is not committed.** It loads the `rag-mcp` working tree. That tree has changes from another session in 4 source files, 5 test files, the README and the new `rag/passage_answers.py`.

## The uncommitted work from 2026-10-02

Purpose (from `docs/specs-2026-10-02-book-references.md`): stop the refusals on Napoleon passage questions, and give short quotes with PDF pages.

1. **Tests:** the 6 related test files pass (88 tests).
2. **Book feature works.** A live query gave a real quote on PDF page 205. This agrees with the spec notes.
3. **Search change is useful.** Search now ignores answer-format sentences, such as "Do not invent details".
4. **False trigger.** Any question with "which … passages" goes to book-quote mode. Live test: "Which function returns the passages to synthesis?" on `rag/server.py` gave "Exact PDF page: unavailable. Printed page: unavailable" and no function name.
5. **Wide trigger.** The words "passages", "source", "pages" and "references" added references to code answers. Test, 7 code questions, trigger on and off: 7 of 7 correct each time, but 7 of 7 "on" answers showed references that nobody requested. One labelled a Python file "PDF page 1". Commit `cc5f173` narrows the trigger. After it, 7 of 7 are correct with no references.
6. **Follow-up questions:** search results do not change. Each search part now also contains a label, such as "Previous question (context only, do not answer)".
7. **Context notes are not checked.** The code checks each quote against the source, but not the context note. Two runs gave two attributions for the same quote on PDF page 205: "Moscow, 1812" and "Eugène Beauharnais, 1805". The page does not settle the attribution, and the second one comes after a section heading.

Verdict: keep it. Commit `ff4004c` keeps the work. Commit `cdd6922` corrects item 4, and commit `cc5f173` corrects item 5. Item 7 is open.

## Status at the end of 2026-10-04

All 873 tests pass. Commits are local on `fix/book-references`.

Corrected:

- The full rebuild after a workspace edit (`5639a6c`).
- Answer length and timeouts from chat (`2cce7f3`).
- Pages for `get_workspaces` (`337e476`).
- Embedding batches that the trace did not show (`9102346`).
- The old scope test (`68bad68`).
- The reason for a refusal (`2d744c8`).

Open: the tool does not check context notes. The next edit of `elpis-core` moves it to the shared document folder, and the next query then builds its index one more time.
