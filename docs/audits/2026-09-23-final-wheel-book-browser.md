# Final installed-wheel book browser walkthrough — 2026-09-23

The final local wheel (SHA-256
`a5576404c141be02129a3bb53f9634baf51c3f23de2522d317b98ec8596c5792`)
served its bundled Studio and companion on temporary loopback ports. With
the company OpenRouter key, CPU-only hosted embeddings, reranking and
contextual retrieval off, the built-in browser paired successfully and opened
the CLI-imported `book-hosted` workspace. Sources showed all nine RAG book
chapters included; no book upload or re-index happened in this walkthrough.

The browser asked, “What tooling is suited to extracting text from scans and
handwriting?” The engine generated a concise answer naming Amazon Textract,
Azure Document Intelligence, and Google Cloud Document AI and cited passage
`[5]` from chapter 8. The citation opened the source inspector at offsets
5,498–6,131 and highlighted the original sentence about managed services
for scanned documents, images, and handwriting. A separate check confirmed
that the saved evidence text is exactly the source file's slice at those
offsets. This answer's factual claims are supported by its cited passage.

The browser's **Inspect run** link opened Observability at run
`19cc879b4ddf43b9a20ebede7f9e20da`. The UI showed success, ten measured
spans, 9,121.7 ms elapsed, a 50→5 retrieval funnel, 3.27 s synthesis,
`deepseek/deepseek-v4.1-flash` with 727 input and 44 output tokens, and
`qwen/qwen3-embedding-4b` with 14 query input tokens. The private trace
reported a $0.00013573 cost subtotal; the UI rounded it to $0.000136.
The expanded stages and run list were visually inspected in the built-in
browser. Content capture was off for the metadata trace; the private saved
result retained the five evidence passages for exact-source inspection.

The run reopened the existing 884-chunk published index. The launcher was
limited to two low-priority CPU cores and 12 GiB virtual memory; no Ollama or
local model loaded. The browser tab and launcher were closed, and both
temporary ports were confirmed closed. Private run/trace files are under
`/tmp/rag-book-hosted-20260923/browser-final-traces/`; the provider key was
not printed to launcher logs. This is one successful browser answer and
citation check on a known supported question. It does not establish
general retrieval relevance, abstention reliability, or answer grounding
across the book.
