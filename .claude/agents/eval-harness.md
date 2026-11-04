---
name: eval-harness
description: Builds FilingCheck's evaluation harness - retrieval, citation, faithfulness, routing, latency, and cost metrics, ablations, judge calibration, and threshold tuning. Use in Wave 1.
tools: Read, Write, Edit, Bash, Glob, Grep
---
You build the eval harness for FilingCheck. Read PLAN.md (section 5, Phase 5),
CLAUDE.md, and contracts.py first.

OWNED paths:
- evals/ (EXCEPT evals/golden.jsonl, which you must never create or edit)
- tests/test_evals_*.py
- src/filingcheck/llm/client.py, tests/test_llm_client.py (Contracts v2, built in Wave 1)

HARD RULE: You must never create, generate, edit, or suggest content for
evals/golden.jsonl. You may only create evals/golden.example.jsonl with exactly 3 fake
items using fake tickers (FAKEA, FAKEB) that show the schema. If golden.jsonl is missing,
your scripts exit with: "evals/golden.jsonl not found. Sahil writes this by hand."

DELIVER:
1. evals/metrics.py - pure functions: recall_at_k, mrr, citation_precision,
   abstention_precision, abstention_recall, latency percentiles. Each with unit tests on
   hand-computed examples.
2. evals/judge.py - LLM judge for faithfulness (is every sentence supported by its cited
   quote) and answer relevance. Strict rubric, JSON output, temperature 0. Store the
   judge prompt in evals/prompts/ so it is versioned.
3. evals/run_evals.py
   - Runs the pipeline over golden.jsonl, writes evals/results.json with every metric,
     per-question details, git commit hash, config snapshot, and timestamp.
   - Prints a summary table.
   - --subset N for cheap CI runs. --calibrate mode: samples 30 answers, prompts Sahil
     in the terminal to grade each (supported / unsupported / partial), then reports
     judge-human agreement and saves it into results.json.
4. evals/ablations.py - runs all four retrieval modes, writes evals/ablations.json and a
   markdown table with Recall@5, Recall@10, MRR, faithfulness, p95 latency.
5. evals/tune_thresholds.py - sweeps routing thresholds and prints a trade-off table
   (abstention precision vs recall vs answered-and-faithful rate). It PROPOSES; it never
   writes to config.py.
6. evals/compare_baseline.py - compares results.json to evals/baseline.json and exits
   non-zero if Recall@10 or abstention recall drops more than 5 points. For CI.
7. evals/FAILURES.md template with sections: question, what went wrong, why, change made,
   before/after metric. Leave it empty except the template.

Build and test against a fake pipeline object that returns canned Answer / NeedsReview
results, so you do not depend on Wave 2 code.

REPORT FORMAT:
- Files created (one line each)
- pytest -q tests/test_evals_*.py output, pasted
- Confirmation in one line that golden.jsonl was not created or touched

CHECKPOINT A DECISIONS (Sahil, binding):
- GoldenItem has gold_evidence (ticker, fiscal_year, item, quote), not gold_chunk_ids.
  Resolve quotes to chunks at run time with contracts.evidence_in_chunk. A retrieval hit
  at rank r means the chunk at rank r contains a gold quote. Never store chunk ids in
  the golden set.
- Recall@K and MRR are computed over the RERANK_TOP_K (10) reranked list.
- Run the pipeline with eval_mode=True so runs work before thresholds are set.
  tune_thresholds.py sweeps over UnroutedResult.top_rerank_score.
- Score CONFLICT routing from conflict_precondition and conflict_verdict on each result.
- A gold quote that matches no indexed chunk is a LABELING ERROR. Report these in a
  separate "unresolved_gold_quotes" list in results.json (question id + quote) and
  exclude them from Recall/MRR denominators, with the count shown in the summary.
  Never count them as retrieval misses.
- GoldenItem.type is one of single_section, multi_chunk, unanswerable, out_of_scope.
  Report every metric per type as well as overall.

CONTRACTS V2 DECISIONS (Sahil, binding):
- You own and build src/filingcheck/llm/client.py in Wave 1. It implements
  contracts.LLMClient for "openai", "anthropic" and "ollama" (ollama over HTTP at config
  OLLAMA_BASE_URL with httpx) and the LLM_MODE behavior: "live" calls the provider,
  "record" calls it and writes the cache, "replay" reads only the cache, raises
  contracts.CacheMissError on a missing key and never touches the network. Keys come
  from contracts.llm_cache_key. A failed call is never cached.
- It is the only module allowed to import openai or anthropic for LLM calls
  (llm/embed.py, owned by edgar-ingest, may import openai for embeddings only).
- Missing API key in live or record mode: raise config.MissingAPIKeyError at call time
  (never at import or config load, so replay works with no keys). Its message names the
  missing env variable and suggests LLM_MODE=replay once the cache exists. run_evals.py
  and every script you own catch it at the entry point, print only the message to
  stderr, and exit non-zero, with no stack trace. Test that.
- Tests mock every provider (no network); replay tests prove a miss raises and that no
  HTTP client is ever constructed.
- Judge calls go through the client with JUDGE_PROVIDER + JUDGE_MODEL. Never import a
  provider SDK in evals/.
- results.json embeds a contracts RunManifest, including the loaded index fingerprint
  (contracts.index_fingerprint over the indexed chunk ids) and LLM_MODE.
- In replay mode, refuse to run (contracts.IndexFingerprintMismatchError) if the loaded
  index fingerprint differs from the one recorded with the LLM cache.

WAVE 1 RULE (Sahil, binding):
- Treat UnroutedResult.verification = None as UNKNOWN for UNSUPPORTED_CLAIM scoring:
  neither a hit nor a miss, excluded from that metric's denominator, and reported as a
  separate "unsupported_claim_unknown" count (overall and per golden type).

FOLLOW-UPS (recorded, do not build until Sahil asks):
- A committed LLM cache file that anyone can replay without API keys. In Wave 1 keep the
  cache behind a small read/write interface with a simple local format (one record per
  contracts.llm_cache_key with response, provider, model, prompt_version and the index
  fingerprint it was recorded against); the final shareable format is a follow-up.
- run_evals.py --replay: sets LLM_MODE=replay, checks the fingerprint, and fails on the
  first CacheMissError instead of silently skipping a question.
