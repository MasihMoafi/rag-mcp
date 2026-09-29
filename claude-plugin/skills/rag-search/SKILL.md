---
name: rag-search
description: Answer questions from local documents (PDF, DOCX, Markdown, code, text) with the rag MCP server. Use when the answer lives in files or folders on disk and grep or reading whole files would be slow or noisy.
---

# rag-search

Use the `query_knowledge_base` tool from the `rag` MCP server.

1. Pick the smallest file or subdirectory likely to hold the answer and pass it as `doc_path`. Never point it at a repository or workspace root by default.
2. Ask a specific question. If the evidence is thin, retry with exact terms or synonyms before widening the scope.
3. For "is X mentioned anywhere" or coverage claims, ask several differently worded questions; one miss is not proof of absence.
4. On a scan-limit error, narrow `doc_path` and continue.
5. Use `extensions` (for example `[".md"]`) to filter a directory by file type.

To inspect or change the pipeline (embedding model, reranker, number of passages), use `get_rag_settings` / `set_rag_settings`, or the `/rag` command. Never put API keys in chat; the user runs `rag setup` in a terminal.
