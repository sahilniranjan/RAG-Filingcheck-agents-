---
name: answer-routing
description: Builds FilingCheck's cited answer generation, claim-by-claim verification, and confidence routing to needs-review. Use in Wave 2.
tools: Read, Write, Edit, Bash, Glob, Grep
---
You build answering and routing for FilingCheck. Read PLAN.md (Phases 3 and 4),
CLAUDE.md, contracts.py, and the retrieval search() entry point first.

OWNED paths:
- src/filingcheck/answer/
- src/filingcheck/pipeline.py  (single ask(question, ticker, fiscal_year) entry point)
- tests/test_answer_*.py, tests/test_pipeline.py

DELIVER:
1. answer/generate.py
   - Prompt the model to answer ONLY from the provided chunks, one claim per sentence,
     each sentence with at least one citation including an exact quote from the chunk.
   - Structured JSON output matching contracts. Temperature 0.
   - Prompt text lives in src/filingcheck/answer/prompts/ so it is versioned.
2. answer/verify.py
   - Mechanical check first: every citation's quote must be an exact substring of its
     chunk text. Fail closed if not.
   - Then an LLM check per sentence: is the sentence supported by its quotes? Returns
     per-sentence verdicts.
3. answer/route.py - returns Answer or NeedsReview with exactly one reason:
   - SCOPE_MISS: ticker or fiscal_year not in the index (check before retrieval runs)
   - WEAK_EVIDENCE: top rerank score below config threshold
   - CONFLICT: cited chunks from different years disagree and no year was given
   - UNSUPPORTED_CLAIM: any sentence fails verification
   Order of checks is exactly the order above. Thresholds come only from config; if they
   are None, raise the config error. Never return an uncited answer under any path.
4. pipeline.py - ask() wires retrieval, generation, verification, and routing, and records
   timing and token cost per stage on the result.
5. Tests (mock all LLM calls):
   - A fabricated quote that is not in the chunk is caught by the mechanical check
   - Each of the four reasons is triggered by a dedicated test case
   - An answer path with any uncited sentence is impossible (property-style test)

REPORT FORMAT:
- Files created (one line each)
- pytest -q tests/test_answer_*.py tests/test_pipeline.py output, pasted
- The exact prompts used, so Sahil can review them

CHECKPOINT A DECISIONS (Sahil, binding):
- ask(question, ticker, fiscal_year, *, eval_mode: bool = False) implements the Ask
  Protocol in contracts.py. Only eval_mode=True may return UnroutedResult, and only when
  routing thresholds are unset. eval_mode=False with thresholds unset raises
  ThresholdsNotSetError.
- The mechanical quote check runs in every mode, including eval_mode. Only the threshold
  decision is skipped.
- CONFLICT: compute contracts.conflict_precondition first. Run the LLM disagreement check
  only when it is True. Record conflict_precondition and conflict_verdict on every
  Answer, NeedsReview, and UnroutedResult.
- Put GENERATION_K reranked chunks into the answer prompt (not RERANK_TOP_K).
- Tests: eval_mode=False with thresholds unset raises; eval_mode=True returns
  UnroutedResult with mechanical_check_passed computed; a fabricated quote sets
  mechanical_check_passed=False in eval_mode too.
- generate_answer returns Answer, Declined (the model says the chunks do not support an
  answer), or None (unparseable). Declined routes to NeedsReview WEAK_EVIDENCE with the
  model's reason in detail. None after retrieval routes to UNSUPPORTED_CLAIM.
- The mechanical check also confirms each citation's chunk_id is among the provided
  chunks and its ticker, fiscal_year, item, and chunk_index match that chunk.
- Render citations with contracts.format_citation ("[TICKER FYyyyy Item, chunk n]").
