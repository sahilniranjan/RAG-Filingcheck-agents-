"""Shared data contracts for FilingCheck.

Every agent and module imports its types from here. These models are frozen
(immutable) and reject unknown fields, so a typo or a drifting schema fails loudly
instead of silently. All models round-trip through JSON with
``Model.model_validate_json(obj.model_dump_json())``.

This file holds types and function signatures, plus four tiny pure helpers
(normalize_whitespace, evidence_in_chunk, conflict_precondition, format_citation)
that define shared rules so every module and test uses one definition. It contains
no feature logic. Change it only through contracts-architect, with a stated reason.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from datetime import date
from enum import StrEnum
from functools import cache
from typing import Annotated, Literal, Protocol, Self, runtime_checkable

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, model_validator

from filingcheck.config import fiscal_year_for

# ---------------------------------------------------------------------------
# Basic value types
# ---------------------------------------------------------------------------

ItemCode = Literal["1A", "1C", "7"]
"""10-K item codes FilingCheck covers: 1A Risk Factors, 1C Cybersecurity, 7 MD&A."""

ITEM_CODES: tuple[ItemCode, ...] = ("1A", "1C", "7")

RetrievalMode = Literal["bm25", "vector", "hybrid", "hybrid_rerank"]
"""Retrieval modes the search() entry point and the eval ablations must support."""

RETRIEVAL_MODES: tuple[RetrievalMode, ...] = ("bm25", "vector", "hybrid", "hybrid_rerank")

ExpectedBehavior = Literal["answer", "needs_review"]

GoldenType = Literal["single_section", "multi_chunk", "unanswerable", "out_of_scope"]
"""The four golden question types from PLAN.md Phase 5."""

GOLDEN_TYPES: tuple[GoldenType, ...] = (
    "single_section",
    "multi_chunk",
    "unanswerable",
    "out_of_scope",
)

Ticker = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9.\-]{0,9}$")]
"""Uppercase ticker symbol, for example AAPL, BRK.B, or FAKEA."""

FiscalYear = Annotated[int, Field(ge=1990, le=2100)]
"""Fiscal year as defined once by config.fiscal_year_for (year the period ends)."""

Cik = Annotated[str, Field(pattern=r"^\d{10}$")]
"""SEC Central Index Key, zero-padded to 10 digits."""

Rank = Annotated[int, Field(ge=1)]
"""Ranks are 1-based: the best result has rank 1."""

Coverage = Mapping[str, frozenset[int]]
"""What is indexed: ticker -> set of fiscal years. Used for SCOPE_MISS checks."""

CHUNK_ID_LENGTH = 32
"""Number of hex characters kept from the sha256 digest for a chunk_id."""

# Gold quote cap. Fixed on purpose so golden item validity never depends on .env or
# chunking settings: a golden.jsonl that validates today validates on every machine.
# config.py refuses to load if CHUNK_OVERLAP_TOKENS is below GOLD_QUOTE_MAX_TOKENS.
GOLD_QUOTE_MAX_TOKENS: int = 60
"""Longest allowed gold evidence quote, in tokens of GOLD_QUOTE_ENCODING."""

GOLD_QUOTE_ENCODING: str = "cl100k_base"
"""tiktoken encoding used to count gold quote tokens. Not read from config."""


class NeedsReviewReason(StrEnum):
    """The single reason a question was routed to needs_review instead of answered.

    Route checks these in exactly this order: SCOPE_MISS, WEAK_EVIDENCE, CONFLICT,
    UNSUPPORTED_CLAIM.
    """

    WEAK_EVIDENCE = "WEAK_EVIDENCE"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    SCOPE_MISS = "SCOPE_MISS"
    CONFLICT = "CONFLICT"


class _Frozen(BaseModel):
    """Base for all contracts: immutable, no extra fields, strict about shape."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=False)


# ---------------------------------------------------------------------------
# chunk_id
# ---------------------------------------------------------------------------


def make_chunk_id(ticker: str, fiscal_year: int, item: str, chunk_index: int, text: str) -> str:
    """Return the stable, deterministic id for a chunk.

    The id is the first 32 hex characters of sha256 over
    "ticker|fiscal_year|item|chunk_index|text" encoded as UTF-8. The same inputs
    always give the same id, on any machine, in any process. Changing the chunk text
    (for example by changing chunk size or overlap) changes the id. That is why the
    golden set labels evidence by quote (GoldEvidence), not by chunk_id.
    """
    key = f"{ticker}|{fiscal_year}|{item}|{chunk_index}|{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:CHUNK_ID_LENGTH]


# ---------------------------------------------------------------------------
# Ingestion contracts
# ---------------------------------------------------------------------------


class Filing(_Frozen):
    """One 10-K filing as listed by SEC EDGAR.

    cik is the 10-digit zero-padded Central Index Key. url points at the primary
    filing document on sec.gov.

    period_of_report is the last day of the period the 10-K covers (reportDate in
    the EDGAR submissions JSON). filed_date is when it was filed, usually weeks
    later. fiscal_year must equal config.fiscal_year_for(period_of_report); the
    model rejects a mismatch, so a filing date can never leak into the fiscal year.
    """

    cik: Cik
    ticker: Ticker
    company: str
    fiscal_year: FiscalYear
    period_of_report: date
    filed_date: date
    accession: str
    url: str

    @model_validator(mode="after")
    def _check_fiscal_year(self) -> Self:
        expected = fiscal_year_for(self.period_of_report)
        if self.fiscal_year != expected:
            raise ValueError(
                f"fiscal_year {self.fiscal_year} does not match period_of_report "
                f"{self.period_of_report.isoformat()} (expected {expected}, see "
                "config.fiscal_year_for)"
            )
        return self


class Section(_Frozen):
    """The cleaned plain text of one item (1A, 1C, or 7) from one filing.

    char_count must equal len(text).
    """

    filing: Filing
    item: ItemCode
    text: str
    char_count: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _check_char_count(self) -> Self:
        if self.char_count != len(self.text):
            raise ValueError(
                f"char_count {self.char_count} does not match len(text) {len(self.text)}"
            )
        return self


class SectionNotFound(_Frozen):
    """Typed "not found" result when an item cannot be located in a filing.

    Extraction returns this instead of guessing. reason is a short plain-English
    explanation, for example "no Item 1C heading found".
    """

    filing: Filing
    item: ItemCode
    reason: str


class Chunk(_Frozen):
    """A section-aware slice of one Section, the unit that is indexed and cited.

    chunk_id must equal make_chunk_id(ticker, fiscal_year, item, chunk_index, text);
    the model rejects a mismatch. Use Chunk.create(...) to have it computed for you.
    chunk_index is 0-based within its section.

    company and cik are stored for display and filtering (they come from the
    Filing). They are not part of the chunk_id hash, so adding them did not change
    any chunk id.
    """

    chunk_id: Annotated[str, Field(min_length=CHUNK_ID_LENGTH, max_length=CHUNK_ID_LENGTH)]
    company: Annotated[str, Field(min_length=1)]
    ticker: Ticker
    cik: Cik
    fiscal_year: FiscalYear
    item: ItemCode
    chunk_index: Annotated[int, Field(ge=0)]
    text: Annotated[str, Field(min_length=1)]
    token_count: Annotated[int, Field(ge=1)]
    filing_url: str

    @model_validator(mode="after")
    def _check_chunk_id(self) -> Self:
        expected = make_chunk_id(
            self.ticker, self.fiscal_year, self.item, self.chunk_index, self.text
        )
        if self.chunk_id != expected:
            raise ValueError(f"chunk_id {self.chunk_id} does not match computed id {expected}")
        return self

    @classmethod
    def create(
        cls,
        *,
        company: str,
        ticker: str,
        cik: str,
        fiscal_year: int,
        item: ItemCode,
        chunk_index: int,
        text: str,
        token_count: int,
        filing_url: str,
    ) -> Chunk:
        """Build a Chunk with its chunk_id computed by make_chunk_id."""
        return cls(
            chunk_id=make_chunk_id(ticker, fiscal_year, item, chunk_index, text),
            company=company,
            ticker=ticker,
            cik=cik,
            fiscal_year=fiscal_year,
            item=item,
            chunk_index=chunk_index,
            text=text,
            token_count=token_count,
            filing_url=filing_url,
        )


# ---------------------------------------------------------------------------
# Retrieval contracts
# ---------------------------------------------------------------------------


class RetrievedChunk(_Frozen):
    """A chunk plus every retrieval stage's view of it.

    Stage fields are None when that stage did not run or did not return this chunk
    (for example vector_rank is None in "bm25" mode). final_rank is the 1-based
    position in the list returned to the caller and is always set.
    """

    chunk: Chunk
    bm25_rank: Rank | None = None
    vector_rank: Rank | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    final_rank: Rank


# ---------------------------------------------------------------------------
# Answer contracts
# ---------------------------------------------------------------------------


class Citation(_Frozen):
    """Evidence for one sentence.

    quote must be an exact, non-empty substring of the text of the chunk named by
    chunk_id, and ticker, fiscal_year, item, and chunk_index must match that chunk.
    The contract cannot check that on its own (it does not hold the chunk);
    answer/verify.py checks it mechanically and fails closed. format_citation gives
    the display form "[TICKER FYyyyy Item, chunk n]".
    """

    chunk_id: Annotated[str, Field(min_length=CHUNK_ID_LENGTH, max_length=CHUNK_ID_LENGTH)]
    ticker: Ticker
    fiscal_year: FiscalYear
    item: ItemCode
    chunk_index: Annotated[int, Field(ge=0)]
    quote: Annotated[str, Field(min_length=1)]


def format_citation(c: Citation) -> str:
    """Return the citation as shown in answers, for example "[FAKEA FY2024 Item 1C, chunk 3]"."""
    return f"[{c.ticker} FY{c.fiscal_year} Item {c.item}, chunk {c.chunk_index}]"


class AnswerSentence(_Frozen):
    """One claim, one sentence, with at least one citation.

    An uncited sentence cannot be constructed: citations has min_length 1.
    """

    text: Annotated[str, Field(min_length=1)]
    citations: Annotated[tuple[Citation, ...], Field(min_length=1)]


class StageMetrics(_Frozen):
    """Timing and token cost for one pipeline stage (retrieve, generate, verify, ...)."""

    stage: str
    latency_ms: Annotated[float, Field(ge=0)]
    prompt_tokens: Annotated[int, Field(ge=0)] = 0
    completion_tokens: Annotated[int, Field(ge=0)] = 0
    cost_usd: Annotated[float, Field(ge=0)] = 0.0


def _check_conflict_fields(conflict_precondition: bool, conflict_verdict: bool | None) -> None:
    """Shared rule: the LLM conflict check runs only when the precondition is true.

    So conflict_verdict must be None whenever conflict_precondition is False.
    """
    if not conflict_precondition and conflict_verdict is not None:
        raise ValueError(
            "conflict_verdict must be None when conflict_precondition is False "
            "(the LLM conflict check only runs when the precondition is true)"
        )


class Answer(_Frozen):
    """A cited answer. Every sentence carries at least one citation.

    stages is filled by pipeline.ask() and may be empty elsewhere.

    conflict_precondition is the deterministic trigger (see conflict_precondition()).
    conflict_verdict is the LLM conflict check result, None if it did not run. An
    Answer cannot carry conflict_verdict True: a detected conflict is never answered
    (fail closed). If the precondition is True the check must have run and found no
    conflict, so conflict_verdict must be False.
    """

    question: str
    sentences: Annotated[tuple[AnswerSentence, ...], Field(min_length=1)]
    status: Literal["answered"] = "answered"
    conflict_precondition: bool = False
    conflict_verdict: bool | None = None
    stages: tuple[StageMetrics, ...] = ()

    @model_validator(mode="after")
    def _check_conflict(self) -> Self:
        _check_conflict_fields(self.conflict_precondition, self.conflict_verdict)
        if self.conflict_verdict is True:
            raise ValueError("an Answer cannot carry a detected conflict; route to CONFLICT")
        if self.conflict_precondition and self.conflict_verdict is None:
            raise ValueError(
                "conflict_precondition is True but the conflict check did not run; "
                "an Answer needs conflict_verdict False"
            )
        return self


class NeedsReview(_Frozen):
    """The question was not answered. Exactly one reason, plus plain-English detail.

    conflict_precondition and conflict_verdict default to False and None, so a
    SCOPE_MISS built before retrieval is valid. reason CONFLICT requires
    conflict_precondition True and conflict_verdict True.
    """

    question: str
    status: Literal["needs_review"] = "needs_review"
    reason: NeedsReviewReason
    detail: str
    conflict_precondition: bool = False
    conflict_verdict: bool | None = None
    stages: tuple[StageMetrics, ...] = ()

    @model_validator(mode="after")
    def _check_conflict(self) -> Self:
        _check_conflict_fields(self.conflict_precondition, self.conflict_verdict)
        if self.reason is NeedsReviewReason.CONFLICT and not (
            self.conflict_precondition and self.conflict_verdict is True
        ):
            raise ValueError(
                "reason CONFLICT requires conflict_precondition True and conflict_verdict True"
            )
        return self


class Declined(_Frozen):
    """Generation declined: the model said the provided chunks do not support an answer.

    reason is the model's own explanation, non-empty. A Declined is not an Answer: it
    does not inherit from Answer, has no method that converts it to one, and its
    status is "declined". Route turns it into NeedsReview WEAK_EVIDENCE with reason
    in detail.
    """

    question: str
    reason: Annotated[str, Field(min_length=1)]
    status: Literal["declined"] = "declined"

    @model_validator(mode="after")
    def _check_reason(self) -> Self:
        if not self.reason.strip():
            raise ValueError("reason must contain non-whitespace text")
        return self


def conflict_precondition(fiscal_year: int | None, citations: Sequence[Citation]) -> bool:
    """Deterministic CONFLICT trigger, checked before any LLM conflict check.

    True only when the question did not pin a fiscal_year (fiscal_year is None) and
    the citations span two or more distinct fiscal years. Empty citations give False.
    """
    if fiscal_year is not None:
        return False
    return len({c.fiscal_year for c in citations}) >= 2


class SentenceVerdict(_Frozen):
    """Verification result for one answer sentence, by its index in Answer.sentences.

    quotes_match is the mechanical exact-substring check. supported is the LLM check;
    it is False whenever quotes_match is False (fail closed).
    """

    sentence_index: Annotated[int, Field(ge=0)]
    quotes_match: bool
    supported: bool
    detail: str = ""

    @model_validator(mode="after")
    def _fail_closed(self) -> Self:
        if self.supported and not self.quotes_match:
            raise ValueError("a sentence whose quotes do not match cannot be supported")
        return self


class VerificationResult(_Frozen):
    """Per-sentence verdicts for one Answer."""

    verdicts: tuple[SentenceVerdict, ...]

    @property
    def all_supported(self) -> bool:
        """True only if there is at least one verdict and every verdict is supported."""
        return bool(self.verdicts) and all(v.supported for v in self.verdicts)


class UnroutedResult(_Frozen):
    """Eval-only output of ask(eval_mode=True) while routing thresholds are unset.

    It records everything the pipeline produced without making the threshold
    decision, so evals can measure scores and later pick thresholds (Checkpoint D).
    It is deliberately not an Answer: it does not inherit from Answer, has no method
    that converts it to one, and its status is "unrouted". It must never be shown to
    a user as an answer.

    It never represents a scope miss. SCOPE_MISS needs no threshold, so ask() returns
    NeedsReview(reason=SCOPE_MISS) for out-of-scope questions in every mode,
    including eval_mode, before retrieval runs. An UnroutedResult exists only for
    questions that passed the scope check.

    draft_sentences may be empty if generation failed. declined_reason is set when
    generation returned Declined, and then draft_sentences must be empty.
    mechanical_check_passed is the exact-substring quote check, which runs in every
    mode. top_rerank_score is None when nothing was retrieved or reranking did not run.

    verification holds the per-sentence verdicts from VerifyAnswer, so evals can
    replay UNSUPPORTED_CLAIM routing offline. It is None when declined_reason is set
    or draft_sentences is empty (there was nothing to verify), and None is also
    allowed when verification did not run. When set, it has exactly one verdict per
    draft sentence, with sentence_index 0..n-1 in order.

    stages holds per-stage timing, tokens, and cost (StageMetrics), the same shape
    Answer and NeedsReview use.
    """

    question: str
    retrieved: tuple[RetrievedChunk, ...]
    draft_sentences: tuple[AnswerSentence, ...] = ()
    declined_reason: str | None = None
    verification: VerificationResult | None = None
    top_rerank_score: float | None
    mechanical_check_passed: bool
    conflict_precondition: bool = False
    conflict_verdict: bool | None = None
    stages: tuple[StageMetrics, ...] = ()
    status: Literal["unrouted"] = "unrouted"

    @model_validator(mode="after")
    def _check_conflict(self) -> Self:
        _check_conflict_fields(self.conflict_precondition, self.conflict_verdict)
        if self.declined_reason is not None and self.draft_sentences:
            raise ValueError("draft_sentences must be empty when declined_reason is set")
        return self

    @model_validator(mode="after")
    def _check_verification(self) -> Self:
        if self.verification is None:
            return self
        if self.declined_reason is not None or not self.draft_sentences:
            raise ValueError(
                "verification must be None when declined_reason is set or "
                "draft_sentences is empty (there is nothing to verify)"
            )
        indexes = [v.sentence_index for v in self.verification.verdicts]
        if indexes != list(range(len(self.draft_sentences))):
            raise ValueError(
                f"verification must have exactly one verdict per draft sentence with "
                f"sentence_index 0..{len(self.draft_sentences) - 1} in order, got {indexes}"
            )
        return self


# ---------------------------------------------------------------------------
# Eval contracts
# ---------------------------------------------------------------------------


_WHITESPACE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse every run of whitespace (spaces, tabs, line breaks) to one space, and strip."""
    return _WHITESPACE.sub(" ", text).strip()


class GoldEvidence(_Frozen):
    """One piece of gold evidence: an exact sentence copied from a filing section.

    quote is copied by hand from the filing text. It is matched against chunks with
    evidence_in_chunk(), which ignores differences in whitespace only.

    quote may be at most GOLD_QUOTE_MAX_TOKENS (60) tokens, counted with tiktoken
    GOLD_QUOTE_ENCODING (cl100k_base). Both are fixed constants in this file, not
    settings: validation never reads env or config, so the same quote is valid or
    invalid on every machine whatever the chunking settings. config.py refuses to
    load if CHUNK_OVERLAP_TOKENS is below the cap, so a quote always fits whole
    inside at least one chunk and never gets cut by a chunk boundary.
    """

    ticker: Ticker
    fiscal_year: FiscalYear
    item: ItemCode
    quote: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def _check_quote(self) -> Self:
        if not normalize_whitespace(self.quote):
            raise ValueError("quote must contain non-whitespace text")
        n_tokens = len(_encoder(GOLD_QUOTE_ENCODING).encode(self.quote))
        if n_tokens > GOLD_QUOTE_MAX_TOKENS:
            raise ValueError(
                f"quote is {n_tokens} tokens, over GOLD_QUOTE_MAX_TOKENS "
                f"{GOLD_QUOTE_MAX_TOKENS}. Pick a shorter sentence, or split it "
                "into two gold_evidence entries."
            )
        return self


@cache
def _encoder(encoding_name: str) -> tiktoken.Encoding:
    """Load a tiktoken encoder once per process and reuse it."""
    return tiktoken.get_encoding(encoding_name)


def evidence_in_chunk(evidence: GoldEvidence, chunk: Chunk) -> bool:
    """Return True if chunk contains this gold evidence. The one shared matching rule.

    True when ticker, fiscal_year, and item all match, and the whitespace-normalized
    quote is a substring of the whitespace-normalized chunk text. Case and punctuation
    must match exactly.
    """
    return (
        evidence.ticker == chunk.ticker
        and evidence.fiscal_year == chunk.fiscal_year
        and evidence.item == chunk.item
        and normalize_whitespace(evidence.quote) in normalize_whitespace(chunk.text)
    )


class GoldenItem(_Frozen):
    """One hand-written eval question from evals/golden.jsonl (written by Sahil only).

    type is one of GOLDEN_TYPES, the four types in PLAN.md Phase 5. ticker and
    fiscal_year are None when the question does not pin them.

    gold_evidence labels the evidence by quote, not by chunk_id. The eval harness
    resolves quotes to chunk ids at run time by whitespace-normalized substring match
    (evidence_in_chunk). A retrieval hit means a returned chunk contains a gold quote.
    This keeps the golden set valid if chunking (size, overlap, tokenizer) changes
    later, because chunk ids change with chunk text but the quotes do not. Quote
    length is checked against the fixed GOLD_QUOTE_MAX_TOKENS cap, never against env
    or chunking settings, so whether an item validates does not depend on .env.

    A gold quote that matches no indexed chunk is a labeling error (a typo, or the
    wrong ticker, year, or item). The eval harness reports those separately; they
    are not counted as retrieval misses.

    Rules: gold_evidence is required and non-empty when expected_behavior is "answer".
    expected_reason is required when expected_behavior is "needs_review" and must be
    None when it is "answer". single_section and multi_chunk items must expect
    "answer"; unanswerable and out_of_scope items must expect "needs_review".
    multi_chunk items need at least 2 gold_evidence entries.
    """

    id: str
    question: str
    type: GoldenType
    ticker: Ticker | None = None
    fiscal_year: FiscalYear | None = None
    gold_evidence: tuple[GoldEvidence, ...] = ()
    expected_behavior: ExpectedBehavior
    expected_reason: NeedsReviewReason | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _check_reason(self) -> Self:
        if self.expected_behavior == "answer" and not self.gold_evidence:
            raise ValueError("gold_evidence is required when expected_behavior is answer")
        if self.expected_behavior == "needs_review" and self.expected_reason is None:
            raise ValueError("expected_reason is required when expected_behavior is needs_review")
        if self.expected_behavior == "answer" and self.expected_reason is not None:
            raise ValueError("expected_reason must be empty when expected_behavior is answer")
        return self

    @model_validator(mode="after")
    def _check_type(self) -> Self:
        wanted: ExpectedBehavior = (
            "answer" if self.type in ("single_section", "multi_chunk") else "needs_review"
        )
        if self.expected_behavior != wanted:
            raise ValueError(f"type {self.type} requires expected_behavior {wanted}")
        if self.type == "multi_chunk" and len(self.gold_evidence) < 2:
            raise ValueError("type multi_chunk requires at least 2 gold_evidence entries")
        return self


# ---------------------------------------------------------------------------
# Function signatures (Protocols). Implementations live in their own modules.
# ---------------------------------------------------------------------------


@runtime_checkable
class FetchFilings(Protocol):
    """edgar: list 10-K filings for a ticker.

    Inputs: ticker (uppercase) and the fiscal years wanted.
    Output: one Filing per fiscal year found, sorted by fiscal_year ascending. Years
    with no 10-K are simply absent from the list.
    Failure: raises an error if SEC_USER_AGENT is unset, if the ticker has no CIK,
    or if EDGAR still fails after retries. Never returns made-up filings.
    """

    def __call__(self, ticker: str, fiscal_years: Sequence[int]) -> list[Filing]: ...


@runtime_checkable
class ExtractSections(Protocol):
    """edgar: pull Items 1A, 1C, and 7 out of one filing's HTML.

    Inputs: the Filing and its raw HTML.
    Output: exactly one entry per item in ITEM_CODES, in that order. Each entry is a
    Section, or a SectionNotFound when the item cannot be located.
    Failure: never guesses. An unlocatable item becomes SectionNotFound; it does not
    raise. Raises only on input that is not parseable HTML at all.
    """

    def __call__(self, filing: Filing, html: str) -> list[Section | SectionNotFound]: ...


@runtime_checkable
class ChunkSection(Protocol):
    """ingest: split one Section into overlapping, token-bounded chunks.

    Inputs: a Section. Chunk size and overlap come from config.
    Output: chunks in order with chunk_index 0..n-1, ids from make_chunk_id. No chunk
    crosses a section boundary. Deterministic: the same Section gives the same chunks.
    Failure: an empty section returns an empty list. Does not raise on normal input.
    """

    def __call__(self, section: Section) -> list[Chunk]: ...


@runtime_checkable
class SearchBM25(Protocol):
    """retrieve: Postgres full-text search with hard ticker and fiscal_year filters.

    Inputs: the question, optional ticker and fiscal_year filters (applied in SQL
    before ranking), and k.
    Output: up to k RetrievedChunk, best first, with bm25_rank and final_rank set and
    other stage fields None.
    Failure: no matches returns an empty list. Database errors propagate.
    """

    def __call__(
        self, question: str, ticker: str | None, fiscal_year: int | None, k: int
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class SearchVector(Protocol):
    """retrieve: pgvector cosine search with the same hard filters as SearchBM25.

    Inputs: the question (embedded inside the call), optional ticker and fiscal_year
    filters, and k.
    Output: up to k RetrievedChunk, best first, with vector_rank and final_rank set
    and other stage fields None.
    Failure: no matches returns an empty list. Embedding or database errors propagate.
    """

    def __call__(
        self, question: str, ticker: str | None, fiscal_year: int | None, k: int
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class RRFMerge(Protocol):
    """retrieve: Reciprocal Rank Fusion of BM25 and vector results. Pure, no I/O.

    Inputs: the two ranked lists and the RRF constant k (config RRF_K, default 60).
    Output: the union by chunk_id, sorted by rrf_score descending, keeping each
    chunk's bm25_rank and vector_rank (None if absent from that list), with
    final_rank renumbered 1..n. Ties are broken by chunk_id ascending so the order
    is deterministic.
    Failure: two empty inputs return an empty list. Does not raise.
    """

    def __call__(
        self,
        bm25: Sequence[RetrievedChunk],
        vector: Sequence[RetrievedChunk],
        k: int,
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class Rerank(Protocol):
    """retrieve: cross-encoder rerank of candidates down to top_k.

    Inputs: the question, candidates (already fused, up to config RERANK_CANDIDATES),
    and top_k (config RERANK_TOP_K, the depth evals measure).
    Output: up to top_k RetrievedChunk sorted by rerank_score descending, keeping all
    earlier stage fields, with rerank_score set and final_rank renumbered 1..n.
    Failure: empty candidates return an empty list. Model load errors propagate.
    """

    def __call__(
        self, question: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class GenerateAnswer(Protocol):
    """answer: draft a cited answer using only the provided chunks.

    Inputs: the question and the retrieved chunks to answer from (the pipeline passes
    the first config GENERATION_K reranked chunks).
    Output: an Answer whose every sentence has at least one Citation pointing at one
    of the provided chunks, or a Declined carrying the model's reason when the model
    says the chunks do not support an answer. Temperature 0.
    Failure: returns None only when the model output cannot be parsed into a valid
    Answer or Declined (for example a sentence with no citation), so routing sends
    it to needs_review. API errors propagate.
    """

    def __call__(
        self, question: str, chunks: Sequence[RetrievedChunk]
    ) -> Answer | Declined | None: ...


@runtime_checkable
class VerifyAnswer(Protocol):
    """answer: check every sentence of a draft Answer against its cited chunks.

    Inputs: the draft Answer and the chunks it was generated from.
    Output: one SentenceVerdict per sentence, in sentence order. The mechanical
    check runs first: each citation's chunk_id must exist among the provided chunks,
    its ticker, fiscal_year, item, and chunk_index must match that chunk, and its
    quote must be an exact substring of that chunk's text. Any mismatch makes that
    sentence unsupported (fail closed).
    The mechanical check always runs, in every mode including eval_mode, and needs no
    routing threshold.
    Failure: if the LLM check errors, the affected sentences are marked unsupported
    rather than passed.
    """

    def __call__(self, answer: Answer, chunks: Sequence[RetrievedChunk]) -> VerificationResult: ...


@runtime_checkable
class Route(Protocol):
    """answer: decide between the Answer and a NeedsReview with exactly one reason.

    Inputs: the question, optional ticker and fiscal_year, the indexed coverage, and
    whatever later stages produced so far (None when a stage has not run yet, which
    lets the pipeline call route for the scope check before retrieval).
    Output: the Answer only if every check passes, otherwise NeedsReview. Checks run
    in this exact order: SCOPE_MISS, WEAK_EVIDENCE, CONFLICT, UNSUPPORTED_CLAIM.
    A Declined draft routes to NeedsReview WEAK_EVIDENCE with the model's reason in
    detail. A None draft after retrieval (output could not be parsed) routes to
    UNSUPPORTED_CLAIM. SCOPE_MISS compares against coverage only and needs no
    threshold.
    Failure: raises config.ThresholdsNotSetError if routing thresholds are None
    before any threshold-based check (WEAK_EVIDENCE onward); Route never skips the
    threshold decision. (Only Ask in eval_mode skips it, by not calling Route for the
    later checks and returning an UnroutedResult.) A scope miss is returned as
    NeedsReview SCOPE_MISS even while thresholds are None. The mechanical quote check
    in VerifyAnswer runs regardless. Never returns an uncited answer.
    """

    def __call__(
        self,
        question: str,
        ticker: str | None,
        fiscal_year: int | None,
        coverage: Coverage,
        retrieved: Sequence[RetrievedChunk] | None,
        draft: Answer | Declined | None,
        verification: VerificationResult | None,
    ) -> Answer | NeedsReview: ...


@runtime_checkable
class Ask(Protocol):
    """pipeline: answer one question end to end (scope check, retrieve, generate,
    verify, route).

    Inputs: the question, optional ticker and fiscal_year, and eval_mode.
    Output: an Answer or a NeedsReview. The scope check runs first, before retrieval,
    in every mode including eval_mode: SCOPE_MISS needs no threshold, so an
    out-of-scope question always returns NeedsReview(reason=SCOPE_MISS), never an
    UnroutedResult. A Declined from generation is never returned as is: it is
    routed, or recorded in UnroutedResult.declined_reason. Only eval_mode=True may
    return an UnroutedResult, only for questions that pass the scope check, and only
    when routing thresholds are unset; with thresholds set, eval_mode=True routes
    normally. The API never exposes eval_mode; it exists for evals/ only. The
    mechanical quote check (every quote is an exact substring of its chunk) runs in
    every mode, including eval_mode; only the threshold decision is skipped.
    Failure: with eval_mode=False and routing thresholds unset, raises
    config.ThresholdsNotSetError (fail closed) for any question that passes the
    scope check. A scope miss is returned as NeedsReview SCOPE_MISS even then. Never
    returns an uncited answer.
    """

    def __call__(
        self,
        question: str,
        ticker: str | None,
        fiscal_year: int | None,
        *,
        eval_mode: bool = False,
    ) -> Answer | NeedsReview | UnroutedResult: ...


__all__ = [
    "CHUNK_ID_LENGTH",
    "GOLDEN_TYPES",
    "GOLD_QUOTE_ENCODING",
    "GOLD_QUOTE_MAX_TOKENS",
    "ITEM_CODES",
    "RETRIEVAL_MODES",
    "Answer",
    "AnswerSentence",
    "Ask",
    "Chunk",
    "ChunkSection",
    "Cik",
    "Citation",
    "Coverage",
    "Declined",
    "ExpectedBehavior",
    "ExtractSections",
    "FetchFilings",
    "Filing",
    "GenerateAnswer",
    "GoldEvidence",
    "GoldenItem",
    "GoldenType",
    "ItemCode",
    "NeedsReview",
    "NeedsReviewReason",
    "RRFMerge",
    "Rerank",
    "RetrievalMode",
    "RetrievedChunk",
    "Route",
    "SearchBM25",
    "SearchVector",
    "Section",
    "SectionNotFound",
    "SentenceVerdict",
    "StageMetrics",
    "UnroutedResult",
    "VerificationResult",
    "VerifyAnswer",
    "conflict_precondition",
    "evidence_in_chunk",
    "format_citation",
    "make_chunk_id",
    "normalize_whitespace",
]
