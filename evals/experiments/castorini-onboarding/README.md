# Castorini retrieval reproduction

Question: do the pinned official Anserini and Pyserini toolkits reproduce the
[Foundations of Retrieval](https://github.com/castorini/onboarding) results?
This measures machine reproduction, not a learner's understanding or admission.

Run only through the Foundations of Retrieval GitHub Actions workflow. The
runner refuses local execution. It uses public MS MARCO and NFCorpus inputs,
the official encoders and published scoring settings. No private corpus,
hosted model API, email, or upstream pull request is involved.

The three jobs cover MS MARCO data preparation and BM25, MS MARCO prebuilt
BM25 and BGE HNSW retrieval, and NFCorpus BGE/BM25 with vector arithmetic.
Expected metrics are set in `run.py` before execution. Raw commands, logs,
dependency versions and ranked runs are retained as workflow artifacts.

SPLADE-v3 requires an approved Hugging Face login. That lesson remains blocked
until access is provided; a different model is not counted as reproducing it.
Status and actual results belong in the workflow artifacts, including failures.
