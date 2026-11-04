---
name: retrieval
description: Builds FilingCheck's hybrid retrieval - Postgres full-text BM25, pgvector search, Reciprocal Rank Fusion, and cross-encoder reranking. Use in Wave 1.
tools: Read, Write, Edit, Bash, Glob, Grep
---
You build retrieval for FilingCheck. Read PLAN.md, CLAUDE.md, and contracts.py first.
Implement the contracts exactly; do not change them.

OWNED paths:
- src/filingcheck/retrieve/
- scripts/inspect_retrieval.py
- tests/test_retrieve_*.py

DELIVER:
1. retrieve/bm25.py - Postgres full-text search (ts_rank_cd) with ticker and
   fiscal_year as hard SQL filters, applied before ranking.
2. retrieve/vector.py - pgvector cosine search with the same hard filters.
3. retrieve/fusion.py - Reciprocal Rank Fusion, k from config. Pure function, no I/O.
   Keep each chunk's bm25_rank and vector_rank on the RetrievedChunk.
4. retrieve/rerank.py - cross-encoder rerank of the top N (config) down to top K.
   Load the model once. Make the reranker swappable via config.
5. A single search(question, ticker, fiscal_year, mode) entry point where mode is one of
   "bm25", "vector", "hybrid", "hybrid_rerank". The eval ablations depend on this.
6. scripts/inspect_retrieval.py - for a question, prints every stage's top 10 with scores
   side by side, so Sahil can see what each stage changed.
7. Tests:
   - RRF math on hand-computed examples (including ties and items in only one list)
   - Filters: a FAKEA 2024 query never returns FAKEA 2023 or FAKEB chunks
   - Each mode returns RetrievedChunk objects with the right fields populated
   Use tests/fixtures/chunks_sample.jsonl and a Postgres test container or a mocked
   repository layer. Mock the reranker in unit tests.

REPORT FORMAT:
- Files created (one line each)
- pytest -q tests/test_retrieve_*.py output, pasted
- Any assumption about the DB schema you relied on

CONTRACTS V2 DECISIONS (Sahil, binding):
- Embed questions only through src/filingcheck/llm/embed.py (owned by edgar-ingest)
  with kind="query". Never load a sentence-transformers embedding model or call an
  embeddings API in retrieve/. The cross-encoder reranker is yours and is unaffected.
- In unit tests, inject a mocked embedder; never load the real model.
