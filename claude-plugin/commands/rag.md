---
description: Change or inspect the RAG answer pipeline in plain words (answer passages, reranker, embedding model)
---
The user wants to change or inspect their RAG answer pipeline: $ARGUMENTS

Use the `rag` command line tool through the shell (if `rag` is not on PATH, use `uvx ragmcp` with the same arguments). Map the request to exactly one of these and run it:
- `rag config show`
- `rag config top <N>`                       (also for "top5", "top 5", "use 5 passages for the answer")
- `rag config reranker off|auto|fastembed|openrouter|<model>`
- `rag config embedding <model> [--provider fastembed|ollama|openai_compatible|sentencetransformer]`
- `rag setup --auto`                          (start over with what this machine already has)
- `rag doctor`                                (what is installed and running)
Never ask for or paste API keys in this chat; for keys, tell the user to run `rag setup` in a terminal.
Finish by running `rag config show` and report its lines to the user, nothing more.
