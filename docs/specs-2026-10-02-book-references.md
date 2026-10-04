# Book references in answers

The Napoleon requests returned four refusals despite relevant book text. Search treated answer format instructions as separate questions. Synthesis received no page data. Its default instruction prohibited source references.

## Required behavior

- Search the content question. Do not search instructions that only specify the answer format.
- Keep distinct content questions as separate search facets.
- Supply verified PDF page ranges when the user requests source references.
- Distinguish PDF page positions from printed page numbers. Do not invent missing page numbers or section names.
- Allow requested short quotes and readable references within the answer string.
- Use the existing synthesis call to select passages. Check each quote against source text before returning it.
- Keep a supported selection when another selection contains a changed quote. State when fewer passages qualify.
- Keep narrator text distinct from the subject's own speech. Preserve the direction of advice.
- Keep the default answer concise. Keep strict support rules and the public response shape.
- Report insufficient evidence with the existing refusal. Keep provider errors separate.
- Report invalid quotes as answer errors. Do not report them as provider access errors.

## Checks

First reproduce the format defect with a failing test. Then test format instructions, distinct questions, available pages, and missing pages. Repeat the four original Napoleon requests through stdio MCP. Check each answer against its PDF pages. An unsupported question must return the exact existing refusal.

Use the existing providers, models, and index. Keep the original comparison evidence. Do not publish changes. Each edit must support this repair.

## Observed checks — 2026-10-02

The user restarted Elpis. The connected MCP process loaded the current server source.
All four original Napoleon questions returned selections through this connection. Six quoted anchors matched their claimed PDF pages and the ten-word limit.
Two questions returned one selection rather than two. Both answers stated the shortfall.
Two unsupported questions returned the exact refusal. A direction check preserved Napoleon's advice to persevere on PDF page 205.
The relevant unit and integration checks passed: 189 tests. These checks do not establish quality for all questions or books.
