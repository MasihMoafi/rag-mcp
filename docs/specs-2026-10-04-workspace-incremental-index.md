# SPEC: Incremental index for workspace edits

## Goal

A workspace edit must embed only the added or changed documents. On 2026-10-04, 4 new files in `elpis-core` caused a full rebuild: 12,748 chunks, 254 s and a new 167 MB index. Other queries waited for it.

## Cause (measured)

- Each edit copies all documents into a new folder, `generations/<revision>/documents` (`rag/workspaces.py:131`).
- The index name comes from that folder and from the list of selected files (`rag/index_lifecycle.py:19`). Thus each edit gets a new index, and the incremental refresh (`rag/ingestion.py:183`) never runs for a workspace.
- No code removes old document copies or old index generations.

## Design

1. Store each document one time in `workspaces/<id>/documents/`. Its file name is its content SHA-256. A file never changes after the write. A revision keeps only its manifest.
2. One store method gives the path of a document. The 5 readers use it: `execution.py:180`, `workspaces.py:215` and `:239`, `chunk_preview.py:47`, `index_inspection.py:90`.
3. A query on all documents of a workspace uses one index name for each workspace and embedding model. The file list goes only into the source manifest. Then the current LanceDB delta code finds the added and removed files.
4. Migration: the first read of an old workspace copies its current documents into the new folder. The old folders stay until Masih accepts the change. The first query after the change builds each workspace index one time.
5. Cleanup: keep the current and the previous revision. Remove the document files and the index generations that only older revisions use.

## Acceptance (write each test first, and show that it fails at the current commit)

- A1 Add: a workspace with 3 documents gets 1 new document. The next query embeds only the chunks of the new document, and it finds the new text.
- A2 Remove: removal of 1 document embeds no text. A query does not find the removed text.
- A3 Change: changed content for the same source path embeds only that document again. A query does not find the old text.
- A4 Same result: after A1, use 3 queries on a full rebuild and on the incremental index. Both give the same top passages, each with the correct `source_id`.
- A5 Storage: after 3 edits, at most 2 index generations stay, and only the document files of the last 2 revisions.
- A6 Concurrency: a query during a save reads one complete revision.
- A7 Live: make a new scratch workspace with 30 files, and add 1 file. The next query embeds only the chunks of that file, in less than 30 s.

## Non-goals

- A query on a subset of sources (`source_ids`) keeps one index for each subset.
- The server runs one query at a time. A build still makes other queries wait.
- The trace shows only the first embedding batch. That needs its own task.
- Qdrant: the delta code supports only LanceDB.

## Rollback

`git revert` of the commits. After a revert, import again each workspace that the new code saved.

## Stop and ask

- Before the removal of the 2 old workspace index folders (332 MB) and the 4 old revision folders.
- If code outside `rag-mcp` reads the revision folders.
- If a test fails after 3 attempts.
- If A4 shows different passages.
