# FilingCheck - Project Plan

An evidence-checking assistant over SEC 10-K filings. Ask a question about a company's disclosures and get an answer backed by cited passages from the actual filing, or an honest "needs review" when the evidence is not strong enough.

Drop this file into the repo root as `PLAN.md` and point Claude Code at it one phase at a time.

**Version 1.1 (Checkpoint A).** Updated with the decisions approved at Checkpoint A: quote-based gold evidence, RERANK_TOP_K and GENERATION_K, eval_mode with UnroutedResult, a deterministic CONFLICT precondition, and a separate judge model. Section 11 lists every decision in one place.


## 1. Why this project

- Public version of the BNY problem: verifying claims against financial documents with evidence, without touching anything confidential.
- Shows the things most portfolio RAG projects skip: hybrid retrieval, citations, confidence routing, and a real eval harness running in CI.
- Every number on the resume comes from this public repo, so it can be checked.

## 2. Scope (keep it small and finished)

- **Companies:** 30, chosen across 5 sectors (banks, payments, insurance, big tech, healthcare). 6 each.
- **Years:** FY2023, FY2024, FY2025 10-Ks. About 90 filings.
- **Sections indexed:** Item 1A (Risk Factors), Item 1C (Cybersecurity, required since FY2023), Item 7 (MD&A). Skip the rest for v1.
- **Out of scope for v1:** 10-Qs, 8-Ks, tables and XBRL numbers, multi-hop questions across companies, chat history.

A finished small project beats an unfinished big one. Expand only after v1 ships with results.

## 3. Architecture

```
EDGAR API -> ingest -> section split -> chunk -> embed -> Postgres (pgvector + full-text)
                                                              |
question -> BM25 search + vector search -> RRF merge -> cross-encoder rerank (top 10)
                                                              |
                         top GENERATION_K chunks -> answer with citations -> confidence check
                                                              |
                                  answer  OR  "needs review" + reason
```

**Stack:** Python 3.11, FastAPI, PostgreSQL 16 with pgvector, OpenAI embeddings (text-embedding-3-small, 1536 dimensions), an open cross-encoder for reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`), Streamlit front end, pytest, GitHub Actions, Docker Compose for local Postgres.

**Models:** generation, verification, and judge are separate config settings. The judge should come from a different model family than the generator, to reduce self-preference bias. Sahil sets the final model names.

## 4. Repo layout

```
filingcheck/
  PLAN.md
  CLAUDE.md                 # conventions for Claude Code (see section 10)
  docker-compose.yml        # postgres + pgvector
  pyproject.toml
  src/filingcheck/
    contracts.py            # shared data models and Protocols (frozen after Checkpoint A)
    config.py               # env vars, thresholds, model names, fiscal year rule
    edgar/client.py         # EDGAR API calls, rate limiting, User-Agent
    edgar/sections.py       # pull Item 1A, 1C, 7 out of the filing HTML
    ingest/chunk.py         # section-aware chunking with overlap
    ingest/load.py          # embed + write to Postgres
    retrieve/bm25.py        # Postgres full-text search
    retrieve/vector.py      # pgvector search
    retrieve/fusion.py      # Reciprocal Rank Fusion
    retrieve/rerank.py      # cross-encoder rerank
    answer/generate.py      # prompt + citation format
    answer/verify.py        # claim-by-claim evidence check
    answer/route.py         # confidence decision + reason
    pipeline.py             # ask(question, ticker, fiscal_year, *, eval_mode=False)
    api/main.py             # FastAPI: POST /ask
  app/streamlit_app.py
  evals/
    golden.jsonl            # hand-written questions (you write these)
    run_evals.py            # runs every metric, writes results.json + table
    ablations.py            # BM25 only vs vector only vs hybrid vs hybrid+rerank
  tests/
  .github/workflows/evals.yml
```

## 5. Build phases

Each phase ends with something that runs. Do not start the next phase until the acceptance check passes.

### Phase 1 - Ingestion (day 1-2)

**Build:** EDGAR client, section extraction, chunking, loading into Postgres.

**EDGAR notes:**
- SEC requires a descriptive User-Agent with your name and email on every request, and asks you to stay under 10 requests per second. Build both into the client from the start.
- Ticker to CIK mapping: `https://www.sec.gov/files/company_tickers.json`
- Filing list per company: `https://data.sec.gov/submissions/CIK##########.json` (CIK zero-padded to 10 digits). Filter `form == "10-K"`.
- Cache every raw download to disk so you never fetch the same filing twice.

**Section extraction is the hard part.** 10-K HTML is messy and every company formats it differently. Find Item headings with regex over the cleaned text, then sanity-check lengths. Log every filing where a section is missing or suspiciously short, and fix those by hand rather than chasing a perfect parser.

**Chunking:** split within a section only, never across sections. 400-600 tokens per chunk with 75 tokens of overlap (about 15%), counted with the cl100k_base tokenizer. Store `company, ticker, cik, fiscal_year, item, chunk_index, text, filing_url`.

**Acceptance:** 90 filings ingested, a coverage report showing which sections were found per filing, and a quick SQL count by company and year.

### Phase 2 - Retrieval (day 3-4)

**Build:** BM25 via Postgres full-text search, vector search via pgvector, RRF merge (k=60), cross-encoder rerank of the top 30 candidates down to RERANK_TOP_K = 10. Evals measure those 10. Only the top GENERATION_K = 6 go into the answer prompt, so retrieval quality and prompt size are tuned separately.

**Filters:** company and fiscal year are filters, not search terms. If the question names a company and year, apply them before searching. A wrong-year answer is a failure even if the text matches.

**Acceptance:** a script that prints the top 10 chunks for 10 sample questions, with scores from each stage, so you can see what each step changed.

### Phase 3 - Answering with citations (day 5)

**Build:** a prompt that answers only from the retrieved chunks and cites every sentence as `[TICKER FY Item, chunk n]`. If the chunks do not support an answer, the model must say so.

**Acceptance:** every sentence in an answer carries a citation that resolves to a real chunk. Write a test that fails if any sentence is uncited.

### Phase 4 - Confidence routing (day 6)

This is the piece that makes the project different. Route to "needs review" when any of these fire, and return which one fired:

1. **Weak evidence:** the top rerank score is below a threshold.
2. **Unsupported claims:** a second LLM pass checks each sentence against its cited chunk. Any sentence judged unsupported fails the answer.
3. **Scope miss:** the company or year in the question is not in the index.
4. **Conflict:** checked in two steps. First a deterministic precondition: the question names no fiscal year AND the cited chunks span 2 or more fiscal years. Only if that is true does an LLM check whether those chunks actually disagree. Both the precondition and the verdict are recorded on every result so evals can score them.

Check order is fixed: scope miss, weak evidence, conflict, unsupported claims. Each result carries exactly one reason.

**The threshold order problem.** Routing needs a threshold, but thresholds are picked from eval data, and evals need the pipeline to run. The fix:
- `ask()` has `eval_mode: bool = False`. With `eval_mode=True` and thresholds unset, it returns an `UnroutedResult`: the draft answer, citations, top rerank score, and conflict fields, with no routing decision. It is not an `Answer` and cannot be turned into one.
- The API never exposes `eval_mode`. While thresholds are unset, `POST /ask` fails closed with a clear error.
- The mechanical quote check (every quote is an exact substring of its chunk) runs in every mode, including eval mode.
- `tune_thresholds.py` applies candidate thresholds to the recorded scores, so threshold tuning never needs the pipeline to make the decision itself.

Pick thresholds from the eval data in Phase 5, not by gut. Record the chosen values in `config.py` with a comment on how they were picked.

**Acceptance:** `/ask` returns either an `Answer` (`{question, sentences: [{text, citations}], status: "answered"}`, citations nested per sentence so each sentence can be checked) or `{status: "needs_review", reason, detail}`. Never an uncited answer. With thresholds unset, `/ask` returns an error and `ask(eval_mode=True)` returns an `UnroutedResult`.

### Phase 5 - Eval harness (day 7-9)

**Golden set: you write this by hand. Do not have an LLM generate it.** That is what makes the numbers trustworthy and what you will be asked about in interviews. Target 80 questions:

| Type | Count | Example | Correct behavior |
|-|-|-|-|
| Answerable, single section | 40 | "What does JPMorgan list as its main cybersecurity governance structure in FY2024?" | Answer, cite the right Item 1C chunk |
| Answerable, needs 2+ chunks | 15 | "How did Visa describe changes in cross-border volume in FY2024 MD&A?" | Answer, cite all supporting chunks |
| Unanswerable | 15 | "What was Capital One's 2024 data center count?" (not disclosed) | Needs review |
| Out of scope or trick | 10 | Wrong year, company not indexed, false premise | Needs review with the right reason |

For each question store: `id, question, type, ticker, fiscal_year, gold_evidence, expected_behavior, expected_reason, notes`.

**Gold evidence is quotes, not chunk ids.** `gold_evidence` is a list of `{ticker, fiscal_year, item, quote}`, where the quote is an exact sentence copied from the filing. It is required and non-empty when `expected_behavior` is `answer`. `expected_reason` is required when `expected_behavior` is `needs_review`, and empty otherwise. The harness maps quotes to chunks at run time by whitespace-normalized substring match, and a retrieval hit means a returned chunk contains a gold quote. This keeps the golden set valid if chunking changes later, and it makes writing it easier: copy sentences straight from the 10-K.

**Metrics:**
- Retrieval: Recall@5, Recall@10, MRR over the reranked top 10, using `gold_evidence` quotes resolved to chunks
- Citations: citation precision (cited chunks that are actually relevant)
- Answer quality: faithfulness and answer relevance via an LLM judge
- Routing: abstention precision and recall (does it flag the unanswerable ones without flagging good ones), plus accuracy per reason
- Evals run with `eval_mode=True` and apply thresholds to recorded scores
- Cost and speed: p50 and p95 latency, cost per query

**Calibrate the judge.** Grade 30 answers by hand and compare to the LLM judge. Report the agreement rate in the README. If agreement is poor, fix the judge prompt before trusting any faithfulness number.

**Ablation table** (this is the table that goes in the README and gets you interview questions):

| Setup | Recall@10 | MRR | Faithfulness | p95 latency |
|-|-|-|-|-|
| BM25 only | | | | |
| Vector only | | | | |
| Hybrid (RRF) | | | | |
| Hybrid + rerank | | | | |

**Acceptance:** `python evals/run_evals.py` writes `results.json` and prints the table above with real numbers.

### Phase 6 - CI, demo, README (day 10-11)

- GitHub Actions runs unit tests on every push, and runs a 20-question eval subset nightly (or on a manual trigger) to control API cost. Fail the build if Recall@10 or abstention recall drops more than 5 points from the last saved baseline.
- Streamlit app: question box, company and year filters, answer with clickable citations that show the source passage, and a visible "needs review" state with its reason.
- Deploy the demo (Streamlit Community Cloud works) and put the link at the top of the README.

**README order:** one-line pitch, demo link and GIF, the results table, architecture diagram, how routing works, how the golden set was built, known limitations, how to run locally.

## 6. Failure log (start this on day 1)

Keep `evals/FAILURES.md`. Every time something breaks in a way you learn from, add one entry: what the question was, what went wrong, why, what you changed, and the before/after metric. Five good entries here are worth more in an interview than the whole codebase, because "walk me through a failure" is the question FDE and AI engineer interviews always ask.

## 7. Known traps

- **Section parsing will eat time.** Cap it at two days. Hand-fix the worst filings and move on.
- **Year confusion.** Fiscal year vs filing date vs period covered. Pick fiscal year, define it once in `config.py`, and test it.
- **Boilerplate risk factors.** Many companies repeat near-identical text year to year. This makes retrieval look good and answers look right for the wrong year. Include year-specific questions in the golden set to catch it.
- **Judge leniency.** LLM judges tend to grade generously. That is why the hand calibration in Phase 5 is required.
- **Scope creep.** Tables, XBRL, 10-Qs and agents can all wait for v2.

## 8. Timeline

About two weeks of focused work, or three to four weeks alongside the job search:

| Week | Goal |
|-|-|
| 1 | Phases 1-3: data in, retrieval working, cited answers |
| 2 | Phases 4-5: routing and the full eval harness with real numbers |
| 3 | Phase 6: CI, demo, README, failure log polished |

Ship v1 even if some numbers are not impressive. An honest table with a clear failure analysis beats a perfect-looking one.

## 9. What goes on the resume (after it is built)

Fill these in only with numbers from `results.json`. Leave them blank until then.

> **FilingCheck** | GitHub | Live demo
> Built an evidence-checking assistant over [N] SEC 10-K filings that answers only with cited passages and routes low-confidence answers to review, reaching Recall@10 of [X] and abstention recall of [Y] on a hand-built [N]-question golden set (Python, FastAPI, pgvector, hybrid retrieval with reranking).

Second line option, if the ablation shows a real gain:

> Measured each retrieval stage in an ablation and found hybrid search plus reranking improved Recall@10 from [A] to [B] over vector-only search, with evals running in CI to catch regressions.

## 10. CLAUDE.md starter

Put this in `CLAUDE.md` so Claude Code follows the same rules every session:

```
# FilingCheck conventions
- Read PLAN.md before starting any task. Work on one phase at a time.
- Python 3.11, type hints everywhere, small functions, no classes unless they hold state (data models and Protocols in contracts.py are fine).
- Every EDGAR request goes through edgar/client.py (User-Agent + rate limit + disk cache).
- Never generate or edit evals/golden.jsonl. Sahil writes it by hand.
- Never hardcode thresholds. They live in config.py with a comment on how they were chosen.
- Every new module gets a pytest file. Run tests before saying a task is done.
- Writing style in docs and comments: plain English, single hyphens only, no em or en dashes.
- When a phase is done, print the acceptance check result and stop. Do not start the next phase.
```

## 11. Checkpoint A decisions (v1.1)

| Topic | Decision |
|-|-|
| Gold labels | `gold_evidence` quotes resolved to chunks at run time, not `gold_chunk_ids` |
| Golden validation | Evidence required for answer items; `expected_reason` required for needs_review items |
| Retrieval depth | RERANK_CANDIDATES 30, RERANK_TOP_K 10, GENERATION_K 6; config refuses inconsistent values |
| Threshold order | `eval_mode` returns `UnroutedResult`; API never exposes it; `/ask` fails closed while thresholds are unset |
| Quote check | Runs in every mode, including eval mode |
| CONFLICT | Deterministic precondition first, LLM disagreement check second, both recorded |
| Models | Separate generation, verifier, and judge settings; judge from a different family; Sahil picks names |
| Kept as is | 400-600 token chunks, 75 overlap, cl100k_base, text-embedding-3-small at 1536, BM25_K 50, VECTOR_K 50, RRF_K 60, ms-marco-MiniLM-L-6-v2, temperature 0, EDGAR 8 req/s, 5 retries, min_rerank_score None until Checkpoint D |
