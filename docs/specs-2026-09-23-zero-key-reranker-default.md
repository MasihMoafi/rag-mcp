# Zero-key retrieval default

## Problem and contract

A fresh local Studio paired to an existing book workspace reports lexical retrieval available but starts with an OpenRouter reranker configured. With no OpenRouter key, a retrieval-only query fails while opening the index, before BM25 can return passages. The first retrieval must work without provider setup or a paid call.

When no saved or environment reranker choice exists, the effective reranker is `off` unless an OpenRouter key is available through the existing environment variable or a configured OpenRouter embedding endpoint. A key-present fresh profile retains the existing hosted reranker default. An explicit saved or environment reranker choice always wins, including `openrouter` without a key; it is never silently rewritten. An effective `off` choice has no active reranker model in the safe config or metadata trace, even when a suggested or previously saved model ID remains in the settings file. Frozen query settings resolve the same effective choice once per run. Studio names the root answer-model field as a configured setting, since retrieval-only runs never call it. No embedding default, answer model, index format, or hosted reranker behavior changes.

## Verification

- Focused settings tests cover zero-key, key-present, explicitly saved, environment-overridden, and frozen settings cases.
- The same nine-chapter, retrieval-only book question that failed in paired Studio succeeds after the fix with the existing 884-chunk index and no key or model call. Its saved trace reports lexical retrieval without a reranker or provider stage; the browser shows the passage evidence.
- Existing settings/retrieval regression tests and the bounded web build remain green.

The lexical result does not establish paraphrase recall or generated-answer quality. Users who explicitly request an unavailable hosted reranker still get an error rather than a silent fallback.

## Verified 2026-09-23

Before the fix, paired Studio loaded all nine sources but the retrieval-only question failed with a `ValueError` during `index.lookup`; the isolated companion logged a missing OpenRouter key. Trace `4bb3d8cb384f496ba2674a4b66cd3da7` records that failure. The fresh default test also failed before the change. After the change, 45 focused settings, control, execution, and trace tests passed. Web typecheck, targeted ESLint, and seven engine-run tests passed; the new retrieval-only funnel test first failed because the selected count was null.

The final paired browser run `01d1b1ab216f446bbb7f8cc2a725eab5` succeeded in retrieval mode with five passages. The exact precision/recall anchor appeared at rank four, and its source button highlighted the exact passage in the chapter-six original in Persian Studio. The 884-chunk index reopened; no index build, answer, reranker, provider span, or reported model usage occurred. English and Persian Observability displayed a `50 → 5` funnel, the safe revisions, and the answer model explicitly as a configured setting; no inactive reranker model was shown. The CPU-limited Next production build and offline wheel/sdist build passed. The wheel embeds the staged bundle byte-for-byte (build ID `ZbpgHcj52qc71N8hOqj8m`, SHA-256 `ee550f9e3fe93d7685423c3c805f2d6953651777b05c9bdecfaad0fc4353e5f0`). No hosted-quality claim, publication, or deployment follows from this check.
