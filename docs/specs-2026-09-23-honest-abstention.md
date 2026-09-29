# Honest abstention when retrieval misses — 2026-09-23

Trigger: the answer stage receives only a few retained passages from an
indexed corpus. A live book run asked what to do with an uncertain graph
entity match. The corpus contains the expert-review rule, but the five
retrieved passages did not. The model abstained while claiming the *indexed
material* lacked the answer. The prompt and no-passage fallback currently
tell it to make that wider claim.

Contract:

- When the supplied passages do not support an answer, say only that there is
  not enough information to answer this question. Do not assert that the
  whole indexed corpus lacks the answer.
- Check the specific relation and conditions requested. A related passage
  describing a different action cannot support an answer; unsupported
  answers use the exact abstention sentence without a citation.
- Keep supported answers and their citation instructions unchanged.
- The fallback with zero passages makes the same bounded claim and makes no
  provider call.
- CLI/MCP answers remain plain text without retrieval internals or citations;
  Studio answers may cite supporting retained passages.

Verify with a failing prompt/fallback test, adjacent answer-pipeline tests,
and one bounded live replay of the known book retrieval miss. The replay can
prove this model followed the corrected instruction once; it is not a broad
grounding or abstention accuracy estimate.

Outcome: the first bounded replay failed by substituting known-duplicate merge
logic for uncertain matching. The strengthened, relation-specific instruction
then passed one replay of that miss and the 53 adjacent tests. The three-case
follow-up did not meet its preregistered all-three rule because one negative
question allowed an alternate, passage-supported answer. See the
[answer-grounding audit](audits/2026-09-23-book-answer-grounding.md) for the
manual evidence judgments and limits.
