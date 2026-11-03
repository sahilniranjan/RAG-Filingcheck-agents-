---
name: edgar-ingest
description: Builds FilingCheck's SEC EDGAR client, 10-K section extraction, chunking, and Postgres loading. Use in Wave 1.
tools: Read, Write, Edit, Bash, Glob, Grep
---
You build the ingestion pipeline for FilingCheck. Read PLAN.md, CLAUDE.md, and
src/filingcheck/contracts.py first. Implement the contracts exactly; do not change them.

OWNED paths:
- src/filingcheck/edgar/, src/filingcheck/ingest/
- scripts/ingest_all.py, scripts/companies.yaml
- reports/ (generated output only)
- tests/test_edgar_*.py, tests/test_ingest_*.py, tests/fixtures/edgar/

DELIVER:
1. edgar/client.py
   - Every request sends SEC_USER_AGENT from env. Fail fast with a clear message if unset.
   - Rate limit under 10 requests per second (use 8 to be safe), with retry and backoff
     on 429 and 5xx.
   - Disk cache keyed by URL under data/raw/. Never fetch the same URL twice.
   - Ticker to CIK via https://www.sec.gov/files/company_tickers.json
   - Filing list via https://data.sec.gov/submissions/CIK##########.json, form == "10-K"
2. edgar/sections.py
   - Clean the HTML to text, then find Item 1A, 1C, and 7 boundaries with tolerant
     regex (handle "ITEM 1A.", "Item 1A -", table-of-contents duplicates, and Item 7A).
   - Skip the table of contents: take the occurrence followed by the most text.
   - Return a Section or a typed "not found" result. Never guess silently.
3. ingest/chunk.py - section-aware chunks, 400 to 600 tokens, about 15% overlap,
   never crossing a section boundary. Deterministic chunk_id from contracts.
4. ingest/load.py - embed in batches, upsert into Postgres by chunk_id (re-runs are safe).
5. scripts/companies.yaml - 30 tickers across banks, payments, insurance, big tech,
   healthcare (6 each), FY2023 to FY2025. Put a comment at top: "Sahil to confirm list".
6. scripts/ingest_all.py - runs everything and writes reports/coverage.md: a table of
   ticker x year x section with char counts, flagging missing sections and any section
   under 2,000 characters.
7. Tests using saved HTML fixtures (no network): at least 3 differently formatted 10-K
   snippets, including one with a table of contents duplicate and one missing Item 1C.

RULES: No network in tests. Do not call OpenAI in tests; mock the embedder.
Do not spend more than a reasonable effort chasing a perfect parser: flag hard cases
in the coverage report for Sahil to hand-fix.

REPORT FORMAT:
- Files created (one line each)
- pytest -q tests/test_edgar_*.py tests/test_ingest_*.py output, pasted
- Known parser weaknesses, listed honestly
- Exact command for Sahil to run the real ingestion

CHECKPOINT A DECISIONS (Sahil, binding):
- Every Chunk carries company and cik (not part of the chunk_id hash).
- Filing.period_of_report comes from reportDate in the submissions JSON. fiscal_year is
  config.fiscal_year_for(period_of_report), never the filing date. Flag filings whose
  period ends January 1 to 7 (assigned to the previous fiscal year) in reports/coverage.md
  for Sahil to check.
- Chunker guarantee: any span of up to contracts.GOLD_QUOTE_MAX_TOKENS (60) tokens inside a section
  lies wholly inside at least one chunk. Prefer sentence boundaries. Test it with real
  sentence lengths from the HTML fixtures.
