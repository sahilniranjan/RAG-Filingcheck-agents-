"""Tests for the shared contracts, config, schema coupling, and fixture chunks."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest
import tiktoken
from pydantic import BaseModel, ValidationError

from filingcheck import config
from filingcheck.contracts import (
    CHUNK_ID_LENGTH,
    GOLD_QUOTE_ENCODING,
    GOLD_QUOTE_MAX_TOKENS,
    GOLDEN_TYPES,
    ITEM_CODES,
    RETRIEVAL_MODES,
    Answer,
    AnswerSentence,
    Ask,
    Chunk,
    ChunkSection,
    Citation,
    Declined,
    ExtractSections,
    FetchFilings,
    Filing,
    GenerateAnswer,
    GoldenItem,
    GoldEvidence,
    NeedsReview,
    NeedsReviewReason,
    Rerank,
    RetrievedChunk,
    Route,
    RRFMerge,
    SearchBM25,
    SearchVector,
    Section,
    SectionNotFound,
    SentenceVerdict,
    StageMetrics,
    UnroutedResult,
    VerificationResult,
    VerifyAnswer,
    conflict_precondition,
    evidence_in_chunk,
    format_citation,
    make_chunk_id,
    normalize_whitespace,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "chunks_sample.jsonl"
SCHEMA = ROOT / "sql" / "schema.sql"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _filing() -> Filing:
    return Filing(
        cik="0000000001",
        ticker="FAKEA",
        company="FAKEA Corp (fictional)",
        fiscal_year=2024,
        period_of_report=date(2024, 12, 31),
        filed_date=date(2025, 2, 14),
        accession="0000000001-25-000001",
        url="https://example.invalid/fake-filings/FAKEA/2024/10-K.htm",
    )


def _chunk(text: str = "FAKEA Corp (fictional) faces interest rate risk.") -> Chunk:
    return Chunk.create(
        company="FAKEA Corp (fictional)",
        ticker="FAKEA",
        cik="0000000001",
        fiscal_year=2024,
        item="1A",
        chunk_index=0,
        text=text,
        token_count=10,
        filing_url="https://example.invalid/fake-filings/FAKEA/2024/10-K.htm",
    )


def _citation(chunk: Chunk) -> Citation:
    return Citation(
        chunk_id=chunk.chunk_id,
        ticker=chunk.ticker,
        fiscal_year=chunk.fiscal_year,
        item=chunk.item,
        chunk_index=chunk.chunk_index,
        quote="interest rate risk",
    )


def _answer() -> Answer:
    c = _chunk()
    return Answer(
        question="What rate risk does FAKEA describe?",
        sentences=(
            AnswerSentence(text="FAKEA faces interest rate risk.", citations=(_citation(c),)),
        ),
        stages=(StageMetrics(stage="generate", latency_ms=12.5, prompt_tokens=100),),
    )


def _unrouted() -> UnroutedResult:
    c = _chunk()
    return UnroutedResult(
        question="What rate risk does FAKEA describe?",
        retrieved=(RetrievedChunk(chunk=c, rerank_score=0.8, final_rank=1),),
        draft_sentences=_answer().sentences,
        top_rerank_score=0.8,
        mechanical_check_passed=True,
        stages=(
            StageMetrics(stage="retrieve", latency_ms=3.0),
            StageMetrics(
                stage="generate",
                latency_ms=40.0,
                prompt_tokens=100,
                completion_tokens=20,
                cost_usd=0.0001,
            ),
        ),
    )


def _verification(n: int, *, supported: bool = True) -> VerificationResult:
    return VerificationResult(
        verdicts=tuple(
            SentenceVerdict(sentence_index=i, quotes_match=True, supported=supported)
            for i in range(n)
        )
    )


def _unrouted_verified() -> UnroutedResult:
    data = {**_unrouted().model_dump(), "verification": _verification(1).model_dump()}
    return UnroutedResult.model_validate(data)


def _all_examples() -> list[BaseModel]:
    f = _filing()
    c = _chunk()
    return [
        f,
        Section(filing=f, item="1C", text="abc", char_count=3),
        SectionNotFound(filing=f, item="1C", reason="no Item 1C heading found"),
        c,
        RetrievedChunk(chunk=c, bm25_rank=1, vector_rank=3, rrf_score=0.032, final_rank=1),
        _citation(c),
        _answer(),
        NeedsReview(
            question="q", reason=NeedsReviewReason.SCOPE_MISS, detail="FAKEZ is not indexed"
        ),
        VerificationResult(
            verdicts=(SentenceVerdict(sentence_index=0, quotes_match=True, supported=True),)
        ),
        GoldenItem(
            id="g1",
            question="q",
            type="single_section",
            ticker="FAKEA",
            fiscal_year=2024,
            gold_evidence=(
                GoldEvidence(
                    ticker="FAKEA", fiscal_year=2024, item="1A", quote="interest rate risk"
                ),
            ),
            expected_behavior="answer",
        ),
        GoldenItem(
            id="g2",
            question="q",
            type="out_of_scope",
            expected_behavior="needs_review",
            expected_reason=NeedsReviewReason.SCOPE_MISS,
        ),
        NeedsReview(
            question="q",
            reason=NeedsReviewReason.CONFLICT,
            detail="2023 and 2024 disagree",
            conflict_precondition=True,
            conflict_verdict=True,
        ),
        _unrouted(),
        Declined(question="q", reason="The chunks do not mention data center counts."),
        UnroutedResult(
            question="q",
            retrieved=(),
            declined_reason="The chunks do not mention data center counts.",
            top_rerank_score=None,
            mechanical_check_passed=True,
        ),
        _unrouted_verified(),
    ]


# ---------------------------------------------------------------------------
# Round-trip and immutability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("obj", _all_examples(), ids=lambda o: type(o).__name__)
def test_json_round_trip(obj: BaseModel) -> None:
    restored = type(obj).model_validate_json(obj.model_dump_json())
    assert restored == obj


@pytest.mark.parametrize("obj", _all_examples(), ids=lambda o: type(o).__name__)
def test_dict_round_trip_via_json_module(obj: BaseModel) -> None:
    as_dict = json.loads(obj.model_dump_json())
    assert type(obj).model_validate(as_dict) == obj


def test_models_are_frozen() -> None:
    c = _chunk()
    with pytest.raises(ValidationError):
        c.text = "changed"  # type: ignore[misc]


def test_extra_fields_rejected() -> None:
    data = json.loads(_chunk().model_dump_json())
    data["surprise"] = 1
    with pytest.raises(ValidationError):
        Chunk.model_validate(data)


# ---------------------------------------------------------------------------
# chunk_id
# ---------------------------------------------------------------------------


def test_chunk_id_is_deterministic() -> None:
    a = make_chunk_id("FAKEA", 2024, "1A", 0, "hello")
    b = make_chunk_id("FAKEA", 2024, "1A", 0, "hello")
    assert a == b
    assert len(a) == CHUNK_ID_LENGTH
    assert re.fullmatch(r"[0-9a-f]+", a)


def test_chunk_id_known_value() -> None:
    # Pinned so an accidental change to the hashing scheme fails loudly.
    # Equals sha256(b"FAKEA|2024|1A|0|hello").hexdigest()[:32].
    assert make_chunk_id("FAKEA", 2024, "1A", 0, "hello") == "55ea3373f8b18f0ee27e227c6313cbbe"


@pytest.mark.parametrize(
    "args",
    [
        ("FAKEB", 2024, "1A", 0, "hello"),
        ("FAKEA", 2023, "1A", 0, "hello"),
        ("FAKEA", 2024, "7", 0, "hello"),
        ("FAKEA", 2024, "1A", 1, "hello"),
        ("FAKEA", 2024, "1A", 0, "hello!"),
    ],
)
def test_chunk_id_changes_with_every_input(args: tuple) -> None:
    assert make_chunk_id(*args) != make_chunk_id("FAKEA", 2024, "1A", 0, "hello")


def test_chunk_rejects_wrong_id() -> None:
    data = json.loads(_chunk().model_dump_json())
    data["chunk_id"] = "0" * CHUNK_ID_LENGTH
    with pytest.raises(ValidationError, match="does not match computed id"):
        Chunk.model_validate(data)


# ---------------------------------------------------------------------------
# Enums and literals
# ---------------------------------------------------------------------------


def test_needs_review_reason_values() -> None:
    assert {r.value for r in NeedsReviewReason} == {
        "WEAK_EVIDENCE",
        "UNSUPPORTED_CLAIM",
        "SCOPE_MISS",
        "CONFLICT",
    }
    assert NeedsReviewReason("CONFLICT") is NeedsReviewReason.CONFLICT


def test_item_codes_and_modes() -> None:
    assert ITEM_CODES == ("1A", "1C", "7")
    assert RETRIEVAL_MODES == ("bm25", "vector", "hybrid", "hybrid_rerank")
    with pytest.raises(ValidationError):
        Section(filing=_filing(), item="7A", text="x", char_count=1)  # type: ignore[arg-type]


def test_status_literals() -> None:
    assert _answer().status == "answered"
    nr = NeedsReview(
        question="q",
        reason=NeedsReviewReason.CONFLICT,
        detail="d",
        conflict_precondition=True,
        conflict_verdict=True,
    )
    assert nr.status == "needs_review"
    assert json.loads(nr.model_dump_json())["reason"] == "CONFLICT"
    assert _unrouted().status == "unrouted"
    with pytest.raises(ValidationError):
        NeedsReview(question="q", status="answered", reason="CONFLICT", detail="d")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        NeedsReview(question="q", reason="NOT_A_REASON", detail="d")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_uncited_sentence_impossible() -> None:
    with pytest.raises(ValidationError):
        AnswerSentence(text="No evidence here.", citations=())


def test_empty_answer_impossible() -> None:
    with pytest.raises(ValidationError):
        Answer(question="q", sentences=())


def test_empty_quote_rejected() -> None:
    c = _chunk()
    with pytest.raises(ValidationError):
        Citation(
            chunk_id=c.chunk_id,
            ticker="FAKEA",
            fiscal_year=2024,
            item="1A",
            chunk_index=0,
            quote="",
        )


def test_section_char_count_must_match() -> None:
    with pytest.raises(ValidationError, match="char_count"):
        Section(filing=_filing(), item="1A", text="abcd", char_count=3)


def test_verdict_fails_closed() -> None:
    with pytest.raises(ValidationError):
        SentenceVerdict(sentence_index=0, quotes_match=False, supported=True)
    assert not VerificationResult(verdicts=()).all_supported


def test_golden_item_reason_consistency() -> None:
    with pytest.raises(ValidationError):
        GoldenItem(id="x", question="q", type="unanswerable", expected_behavior="needs_review")
    with pytest.raises(ValidationError):
        GoldenItem(
            id="x",
            question="q",
            type="single_section",
            expected_behavior="answer",
            expected_reason=NeedsReviewReason.CONFLICT,
            gold_evidence=(
                GoldEvidence(ticker="FAKEA", fiscal_year=2024, item="1A", quote="risk"),
            ),
        )


def test_golden_answer_requires_gold_evidence() -> None:
    with pytest.raises(ValidationError, match="gold_evidence"):
        GoldenItem(id="x", question="q", type="single_section", expected_behavior="answer")
    with pytest.raises(ValidationError):
        GoldEvidence(ticker="FAKEA", fiscal_year=2024, item="1A", quote="")
    with pytest.raises(ValidationError):
        GoldEvidence(ticker="FAKEA", fiscal_year=2024, item="1A", quote="  \n ")


def test_ranks_are_one_based() -> None:
    with pytest.raises(ValidationError):
        RetrievedChunk(chunk=_chunk(), bm25_rank=0, final_rank=1)


def test_protocols_are_runtime_checkable() -> None:
    def anything(*args: object, **kwargs: object) -> None:
        return None

    for proto in (
        FetchFilings,
        ExtractSections,
        ChunkSection,
        SearchBM25,
        SearchVector,
        RRFMerge,
        Rerank,
        GenerateAnswer,
        VerifyAnswer,
        Route,
        Ask,
    ):
        assert isinstance(anything, proto)
        assert proto.__doc__ and "Failure" in proto.__doc__


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_default_settings_load_without_env() -> None:
    s = config.load_settings({})
    assert s.rrf_k == 60
    assert (s.bm25_k, s.vector_k) == (50, 50)
    assert (s.rerank_candidates, s.rerank_top_k, s.generation_k) == (30, 10, 6)
    assert (s.generation_model, s.verifier_model, s.judge_model) == (
        "gpt-4o-mini",
        "gpt-4o-mini",
        "gpt-4o",
    )
    assert s.judge_provider == "openai"
    assert s.sec_rate_limit_rps < 10
    assert s.routing.min_rerank_score is None


def test_require_thresholds_raises_when_unset() -> None:
    with pytest.raises(config.ThresholdsNotSetError, match="Checkpoint D"):
        config.require_routing_thresholds(config.load_settings({}))


def test_require_thresholds_when_set_from_env() -> None:
    s = config.load_settings({"ROUTE_MIN_RERANK_SCORE": "0.25"})
    assert config.require_routing_thresholds(s).min_rerank_score == 0.25


def test_empty_env_value_means_unset() -> None:
    s = config.load_settings({"ROUTE_MIN_RERANK_SCORE": "  ", "RRF_K": ""})
    assert s.routing.min_rerank_score is None
    assert s.rrf_k == 60


def test_bad_env_values_raise_config_error() -> None:
    with pytest.raises(config.ConfigError):
        config.load_settings({"RRF_K": "sixty"})
    with pytest.raises(config.ConfigError):
        config.load_settings({"EMBEDDING_DIM": "768"})
    with pytest.raises(config.ConfigError):
        config.load_settings({"SEC_RATE_LIMIT_RPS": "10"})


def test_embedding_dim_matches_schema() -> None:
    s = config.load_settings({})
    assert config.KNOWN_EMBEDDING_DIMS[s.embedding_model] == s.embedding_dim
    dims = re.findall(r"vector\((\d+)\)", SCHEMA.read_text())
    assert dims == [str(s.embedding_dim)]


def test_schema_has_required_indexes() -> None:
    sql = SCHEMA.read_text().lower()
    assert "unique (chunk_id)" in sql
    assert "using gin (tsv)" in sql
    assert "using hnsw (embedding vector_cosine_ops)" in sql


# ---------------------------------------------------------------------------
# Fixture chunks
# ---------------------------------------------------------------------------


def _fixture_chunks() -> list[Chunk]:
    lines = FIXTURE.read_text().splitlines()
    return [Chunk.model_validate_json(line) for line in lines if line.strip()]


def test_fixture_shape() -> None:
    chunks = _fixture_chunks()
    assert len(chunks) == 40
    assert {c.ticker for c in chunks} == {"FAKEA", "FAKEB", "FAKEC", "FAKED"}
    assert {c.fiscal_year for c in chunks} == {2023, 2024}
    assert {c.item for c in chunks} == set(ITEM_CODES)
    assert len({c.chunk_id for c in chunks}) == 40


def test_fixture_is_clearly_fake() -> None:
    for c in _fixture_chunks():
        assert "(fictional)" in c.text
        assert "NOT A REAL FILING" in c.text
        assert c.filing_url.startswith("https://example.invalid/")


def test_fixture_shares_vocabulary_across_filters() -> None:
    chunks = _fixture_chunks()
    with_term = {(c.ticker, c.fiscal_year) for c in chunks if "cybersecurity" in c.text.lower()}
    assert len(with_term) == 8


# ---------------------------------------------------------------------------
# Retrieval depth config
# ---------------------------------------------------------------------------


def test_generation_k_must_not_exceed_rerank_top_k() -> None:
    with pytest.raises(config.ConfigError, match="GENERATION_K"):
        config.load_settings({"GENERATION_K": "11", "RERANK_TOP_K": "10"})
    s = config.load_settings({"GENERATION_K": "10", "RERANK_TOP_K": "10"})
    assert s.generation_k == 10


def test_rerank_top_k_must_not_exceed_candidates() -> None:
    with pytest.raises(config.ConfigError, match="RERANK_CANDIDATES"):
        config.load_settings({"RERANK_TOP_K": "31", "RERANK_CANDIDATES": "30"})


@pytest.mark.parametrize(
    ("old", "new"), [("RERANK_TOP_N", "RERANK_CANDIDATES"), ("VERIFY_MODEL", "VERIFIER_MODEL")]
)
def test_old_env_names_fail_loudly(old: str, new: str) -> None:
    with pytest.raises(config.ConfigError, match=f"{old}.*{new}"):
        config.load_settings({old: "5"})
    # An empty value counts as unset, like every other variable.
    assert config.load_settings({old: "  "}).rerank_candidates == 30


# ---------------------------------------------------------------------------
# UnroutedResult is never an Answer
# ---------------------------------------------------------------------------


def test_unrouted_is_not_an_answer() -> None:
    u = _unrouted()
    assert not isinstance(u, Answer)
    assert not isinstance(u, NeedsReview)
    assert not issubclass(UnroutedResult, Answer)
    assert not hasattr(u, "to_answer") and not hasattr(u, "as_answer")
    with pytest.raises(ValidationError):
        Answer.model_validate(u.model_dump())
    with pytest.raises(ValidationError):
        NeedsReview.model_validate(u.model_dump())
    with pytest.raises(ValidationError):
        Answer.model_validate_json(u.model_dump_json())


def test_unrouted_allows_failed_generation() -> None:
    u = UnroutedResult(
        question="q",
        retrieved=(),
        top_rerank_score=None,
        mechanical_check_passed=False,
    )
    assert u.draft_sentences == ()
    assert u.stages == ()
    with pytest.raises(ValidationError):
        StageMetrics(stage="generate", latency_ms=1.0, prompt_tokens=-1)
    with pytest.raises(ValidationError):
        UnroutedResult(
            question="q",
            retrieved=(),
            top_rerank_score=None,
            mechanical_check_passed=False,
            status="answered",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# CONFLICT: deterministic precondition, then model verdict
# ---------------------------------------------------------------------------


def _cite(year: int) -> Citation:
    c = Chunk.create(
        company="FAKEA Corp (fictional)",
        ticker="FAKEA",
        cik="0000000001",
        fiscal_year=year,
        item="7",
        chunk_index=0,
        text=f"FAKEA Corp (fictional) revenue grew in {year}.",
        token_count=10,
        filing_url=f"https://example.invalid/fake-filings/FAKEA/{year}/10-K.htm",
    )
    return Citation(
        chunk_id=c.chunk_id,
        ticker="FAKEA",
        fiscal_year=year,
        item="7",
        chunk_index=0,
        quote="revenue grew",
    )


def test_conflict_precondition() -> None:
    assert conflict_precondition(2024, [_cite(2023), _cite(2024)]) is False
    assert conflict_precondition(None, [_cite(2024), _cite(2024)]) is False
    assert conflict_precondition(None, [_cite(2023), _cite(2024)]) is True
    assert conflict_precondition(None, []) is False


def test_scope_miss_before_retrieval_is_valid() -> None:
    nr = NeedsReview(question="q", reason=NeedsReviewReason.SCOPE_MISS, detail="not indexed")
    assert nr.conflict_precondition is False
    assert nr.conflict_verdict is None


@pytest.mark.parametrize("verdict", [True, False])
def test_verdict_requires_precondition_everywhere(verdict: bool) -> None:
    with pytest.raises(ValidationError, match="conflict_verdict must be None"):
        Answer(question="q", sentences=_answer().sentences, conflict_verdict=verdict)
    with pytest.raises(ValidationError, match="conflict_verdict must be None"):
        NeedsReview(
            question="q",
            reason=NeedsReviewReason.WEAK_EVIDENCE,
            detail="d",
            conflict_verdict=verdict,
        )
    with pytest.raises(ValidationError, match="conflict_verdict must be None"):
        UnroutedResult(
            question="q",
            retrieved=(),
            top_rerank_score=None,
            mechanical_check_passed=False,
            conflict_verdict=verdict,
        )


@pytest.mark.parametrize(("precondition", "verdict"), [(False, None), (True, None), (True, False)])
def test_conflict_reason_requires_true_precondition_and_verdict(
    precondition: bool, verdict: bool | None
) -> None:
    with pytest.raises(ValidationError, match="reason CONFLICT requires"):
        NeedsReview(
            question="q",
            reason=NeedsReviewReason.CONFLICT,
            detail="d",
            conflict_precondition=precondition,
            conflict_verdict=verdict,
        )


def test_answer_rejects_detected_conflict() -> None:
    sentences = _answer().sentences
    with pytest.raises(ValidationError, match="detected conflict"):
        Answer(question="q", sentences=sentences, conflict_precondition=True, conflict_verdict=True)
    with pytest.raises(ValidationError, match="did not run"):
        Answer(question="q", sentences=sentences, conflict_precondition=True)
    ok = Answer(
        question="q", sentences=sentences, conflict_precondition=True, conflict_verdict=False
    )
    assert ok.conflict_verdict is False


def test_unrouted_may_hold_detected_conflict() -> None:
    u = UnroutedResult(
        question="q",
        retrieved=(),
        top_rerank_score=None,
        mechanical_check_passed=True,
        conflict_precondition=True,
        conflict_verdict=True,
    )
    assert u.conflict_verdict is True


# ---------------------------------------------------------------------------
# Quote-based gold evidence survives a chunking change
# ---------------------------------------------------------------------------

FAKE_SECTION = (
    "NOT A REAL FILING. FAKEA Corp (fictional) sells imaginary widgets. "
    "Demand for widgets is seasonal and peaks in the fourth quarter. "
    "FAKEA Corp (fictional) relies on a single supplier for widget casings. "
    "A disruption at that supplier could delay shipments for several months. "
    "The company holds variable rate debt and is exposed to interest rate changes. "
    "Management reviews these risks with the board each quarter."
)
GOLD_QUOTE = "A disruption at that supplier could delay shipments for several months."


def _split(text: str, size: int, overlap: int) -> list[Chunk]:
    """Toy word-based chunker for this test only: windows of size words, overlapping."""
    words = text.split(" ")
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        piece = " ".join(words[start : start + size])
        chunks.append(
            Chunk.create(
                company="FAKEA Corp (fictional)",
                ticker="FAKEA",
                cik="0000000001",
                fiscal_year=2024,
                item="1A",
                chunk_index=len(chunks),
                text=piece,
                token_count=len(piece.split()),
                filing_url="https://example.invalid/fake-filings/FAKEA/2024/10-K.htm",
            )
        )
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


def _gold_item(quote: str = GOLD_QUOTE) -> GoldenItem:
    return GoldenItem(
        id="g-supplier",
        question="What supplier risk does FAKEA describe?",
        type="single_section",
        ticker="FAKEA",
        fiscal_year=2024,
        gold_evidence=(GoldEvidence(ticker="FAKEA", fiscal_year=2024, item="1A", quote=quote),),
        expected_behavior="answer",
    )


def test_gold_evidence_survives_rechunking() -> None:
    item = GoldenItem.model_validate_json(_gold_item().model_dump_json())
    evidence = item.gold_evidence[0]
    small = _split(FAKE_SECTION, size=20, overlap=12)
    large = _split(FAKE_SECTION, size=40, overlap=15)
    assert {c.chunk_id for c in small}.isdisjoint({c.chunk_id for c in large})
    assert any(evidence_in_chunk(evidence, c) for c in small)
    assert any(evidence_in_chunk(evidence, c) for c in large)


def test_whitespace_normalized_matching() -> None:
    assert normalize_whitespace("  a\n\tb   c \r\n") == "a b c"
    messy = "A disruption at that\nsupplier could   delay\n\tshipments for several months."
    evidence = _gold_item(messy).gold_evidence[0]
    chunk = _split(FAKE_SECTION, size=100, overlap=0)[0]
    assert evidence_in_chunk(evidence, chunk)
    wrapped = Chunk.create(
        company="FAKEA Corp (fictional)",
        ticker="FAKEA",
        cik="0000000001",
        fiscal_year=2024,
        item="1A",
        chunk_index=0,
        text=FAKE_SECTION.replace("could delay", "could\n   delay"),
        token_count=60,
        filing_url="https://example.invalid/fake-filings/FAKEA/2024/10-K.htm",
    )
    assert evidence_in_chunk(_gold_item().gold_evidence[0], wrapped)


@pytest.mark.parametrize("change", [{"ticker": "FAKEB"}, {"fiscal_year": 2023}, {"item": "7"}])
def test_wrong_filters_do_not_match(change: dict) -> None:
    chunk = _split(FAKE_SECTION, size=100, overlap=0)[0]
    base = {"ticker": "FAKEA", "fiscal_year": 2024, "item": "1A", "quote": GOLD_QUOTE}
    assert evidence_in_chunk(GoldEvidence(**base), chunk)
    assert not evidence_in_chunk(GoldEvidence(**{**base, **change}), chunk)
    assert not evidence_in_chunk(GoldEvidence(**{**base, "quote": "not in the text"}), chunk)


# ---------------------------------------------------------------------------
# company and cik on chunks
# ---------------------------------------------------------------------------


def test_chunk_company_and_cik_do_not_change_chunk_id() -> None:
    a = _chunk()
    data = json.loads(a.model_dump_json())
    b = Chunk.model_validate({**data, "company": "Other Name (fictional)", "cik": "0000000009"})
    assert a.chunk_id == b.chunk_id == make_chunk_id("FAKEA", 2024, "1A", 0, a.text)
    with pytest.raises(ValidationError):
        Chunk.model_validate({**data, "cik": "123"})
    without = {k: v for k, v in data.items() if k not in ("company", "cik")}
    with pytest.raises(ValidationError):
        Chunk.model_validate(without)


def test_schema_has_company_and_cik() -> None:
    sql = SCHEMA.read_text().lower()
    assert re.search(r"\bcompany\s+text\s+not null", sql)
    assert re.search(r"\bcik\s+text\s+not null check \(cik ~ '\^\[0-9\]\{10\}\$'\)", sql)


def test_fixture_company_and_cik() -> None:
    by_ticker = {(c.ticker, c.company, c.cik) for c in _fixture_chunks()}
    assert by_ticker == {
        ("FAKEA", "FAKEA Corp (fictional)", "0000000001"),
        ("FAKEB", "FAKEB Payments Inc (fictional)", "0000000002"),
        ("FAKEC", "FAKEC Mutual Holdings (fictional)", "0000000003"),
        ("FAKED", "FAKED Cloud Systems (fictional)", "0000000004"),
    }


# ---------------------------------------------------------------------------
# Citation display format
# ---------------------------------------------------------------------------


def test_format_citation() -> None:
    c = Citation(
        chunk_id="0" * CHUNK_ID_LENGTH,
        ticker="FAKEA",
        fiscal_year=2024,
        item="1C",
        chunk_index=3,
        quote="board oversight",
    )
    assert format_citation(c) == "[FAKEA FY2024 Item 1C, chunk 3]"
    assert format_citation(_citation(_chunk())) == "[FAKEA FY2024 Item 1A, chunk 0]"
    with pytest.raises(ValidationError):
        Citation(**{**c.model_dump(), "chunk_index": -1})


# ---------------------------------------------------------------------------
# Declined generation
# ---------------------------------------------------------------------------


def test_declined_is_not_an_answer() -> None:
    d = Declined(question="q", reason="No chunk mentions a data center count.")
    assert d.status == "declined"
    assert not isinstance(d, Answer)
    assert not issubclass(Declined, Answer)
    assert not hasattr(d, "to_answer") and not hasattr(d, "as_answer")
    with pytest.raises(ValidationError):
        Answer.model_validate(d.model_dump())
    with pytest.raises(ValidationError):
        Answer.model_validate_json(d.model_dump_json())
    with pytest.raises(ValidationError):
        NeedsReview.model_validate(d.model_dump())
    with pytest.raises(ValidationError):
        Declined(question="q", reason="r", status="answered")  # type: ignore[arg-type]


@pytest.mark.parametrize("reason", ["", "   \n"])
def test_declined_needs_a_reason(reason: str) -> None:
    with pytest.raises(ValidationError):
        Declined(question="q", reason=reason)


def test_unrouted_declined_has_no_draft() -> None:
    with pytest.raises(ValidationError, match="draft_sentences must be empty"):
        UnroutedResult(
            question="q",
            retrieved=(),
            draft_sentences=_answer().sentences,
            declined_reason="not supported",
            top_rerank_score=None,
            mechanical_check_passed=True,
        )
    assert _unrouted().declined_reason is None


# ---------------------------------------------------------------------------
# Golden question types
# ---------------------------------------------------------------------------


def _ev(quote: str = "interest rate risk") -> GoldEvidence:
    return GoldEvidence(ticker="FAKEA", fiscal_year=2024, item="1A", quote=quote)


def test_golden_types() -> None:
    assert GOLDEN_TYPES == ("single_section", "multi_chunk", "unanswerable", "out_of_scope")
    with pytest.raises(ValidationError):
        GoldenItem(
            id="x", question="q", type="factual", expected_behavior="answer", gold_evidence=(_ev(),)
        )


@pytest.mark.parametrize("gtype", ["single_section", "multi_chunk"])
def test_answerable_types_require_answer(gtype: str) -> None:
    ok = GoldenItem(
        id="x",
        question="q",
        type=gtype,  # type: ignore[arg-type]
        expected_behavior="answer",
        gold_evidence=(_ev(), _ev("rate risk")),
    )
    assert ok.type == gtype
    with pytest.raises(ValidationError, match="requires expected_behavior answer"):
        GoldenItem(
            id="x",
            question="q",
            type=gtype,  # type: ignore[arg-type]
            expected_behavior="needs_review",
            expected_reason=NeedsReviewReason.WEAK_EVIDENCE,
        )


@pytest.mark.parametrize("gtype", ["unanswerable", "out_of_scope"])
def test_unanswerable_types_require_needs_review(gtype: str) -> None:
    ok = GoldenItem(
        id="x",
        question="q",
        type=gtype,  # type: ignore[arg-type]
        expected_behavior="needs_review",
        expected_reason=NeedsReviewReason.SCOPE_MISS,
    )
    assert ok.type == gtype
    with pytest.raises(ValidationError, match="requires expected_behavior needs_review"):
        GoldenItem(
            id="x",
            question="q",
            type=gtype,  # type: ignore[arg-type]
            expected_behavior="answer",
            gold_evidence=(_ev(),),
        )


def test_multi_chunk_needs_two_evidence_entries() -> None:
    with pytest.raises(ValidationError, match="at least 2 gold_evidence"):
        GoldenItem(
            id="x",
            question="q",
            type="multi_chunk",
            expected_behavior="answer",
            gold_evidence=(_ev(),),
        )


# ---------------------------------------------------------------------------
# Fiscal year is defined once, from the period of report
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        (date(2024, 12, 31), 2024),
        (date(2025, 1, 2), 2024),
        (date(2025, 1, 7), 2024),
        (date(2025, 1, 8), 2025),
        (date(2024, 6, 30), 2024),
        (date(2024, 9, 30), 2024),
    ],
    ids=["december", "jan_2_53_week", "jan_7_last_early_day", "jan_8", "msft_june", "v_sept"],
)
def test_fiscal_year_for(period: date, expected: int) -> None:
    assert config.fiscal_year_for(period) == expected


def test_filing_accepts_52_53_week_year_ending_in_early_january() -> None:
    data = json.loads(_filing().model_dump_json())
    f = Filing.model_validate({**data, "period_of_report": "2025-01-02", "fiscal_year": 2024})
    assert f.fiscal_year == 2024
    with pytest.raises(ValidationError, match="does not match period_of_report"):
        Filing.model_validate({**data, "period_of_report": "2025-01-02", "fiscal_year": 2025})


def test_filing_fiscal_year_comes_from_period_not_filed_date() -> None:
    f = _filing()
    assert (f.period_of_report, f.filed_date) == (date(2024, 12, 31), date(2025, 2, 14))
    assert f.fiscal_year == 2024


def test_filing_rejects_fiscal_year_that_disagrees_with_period() -> None:
    data = json.loads(_filing().model_dump_json())
    with pytest.raises(ValidationError, match="does not match period_of_report"):
        Filing.model_validate({**data, "fiscal_year": 2025})
    without = {k: v for k, v in data.items() if k != "period_of_report"}
    with pytest.raises(ValidationError):
        Filing.model_validate(without)


# ---------------------------------------------------------------------------
# Gold quote length cap
# ---------------------------------------------------------------------------


def _quote_with_tokens(n: int) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    quote = "risk" + " risk" * (n - 1)
    assert len(enc.encode(quote)) == n
    return quote


def test_gold_quote_cap_is_a_fixed_constant() -> None:
    assert GOLD_QUOTE_MAX_TOKENS == 60
    assert GOLD_QUOTE_ENCODING == "cl100k_base"
    assert not hasattr(config.load_settings({}), "gold_quote_max_tokens")


def test_gold_quote_token_cap() -> None:
    assert _ev(_quote_with_tokens(60)).quote
    with pytest.raises(ValidationError, match="Pick a shorter sentence, or split it"):
        _ev(_quote_with_tokens(61))


@pytest.mark.parametrize(
    "env",
    [
        {"GOLD_QUOTE_MAX_TOKENS": "10"},
        {"GOLD_QUOTE_MAX_TOKENS": "500"},
        {"CHUNK_OVERLAP_TOKENS": "200", "CHUNK_MIN_TOKENS": "300"},
        {"TOKENIZER_ENCODING": "o200k_base"},
        {"CHUNK_MAX_TOKENS": "1000"},
    ],
)
def test_gold_quote_validity_ignores_env(monkeypatch: pytest.MonkeyPatch, env: dict) -> None:
    ok, too_long = _quote_with_tokens(60), _quote_with_tokens(61)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    config.get_settings.cache_clear()
    try:
        assert _ev(ok).quote == ok
        with pytest.raises(ValidationError, match="over GOLD_QUOTE_MAX_TOKENS 60"):
            _ev(too_long)
    finally:
        monkeypatch.undo()
        config.get_settings.cache_clear()


def test_chunk_overlap_must_hold_a_gold_quote() -> None:
    with pytest.raises(config.ConfigError, match="must fit whole inside a chunk overlap"):
        config.load_settings({"CHUNK_OVERLAP_TOKENS": "59"})
    assert config.load_settings({"CHUNK_OVERLAP_TOKENS": "60"}).chunk_overlap_tokens == 60


def test_gold_quote_max_tokens_env_is_refused() -> None:
    with pytest.raises(config.ConfigError, match="GOLD_QUOTE_MAX_TOKENS.*fixed constant"):
        config.load_settings({"GOLD_QUOTE_MAX_TOKENS": "60"})
    assert config.load_settings({"GOLD_QUOTE_MAX_TOKENS": " "}).chunk_overlap_tokens == 75


# ---------------------------------------------------------------------------
# UnroutedResult carries per-sentence verification
# ---------------------------------------------------------------------------


def _two_sentences() -> tuple[AnswerSentence, ...]:
    cite = (_citation(_chunk()),)
    return (
        AnswerSentence(text="FAKEA faces interest rate risk.", citations=cite),
        AnswerSentence(text="FAKEA names rate risk again.", citations=cite),
    )


def _unrouted_with(**kwargs: object) -> UnroutedResult:
    base: dict[str, object] = {
        "question": "q",
        "retrieved": (),
        "top_rerank_score": None,
        "mechanical_check_passed": True,
    }
    return UnroutedResult(**{**base, **kwargs})  # type: ignore[arg-type]


def test_unrouted_verification_defaults_to_none() -> None:
    assert _unrouted().verification is None
    assert _unrouted_with(draft_sentences=_two_sentences()).verification is None


def test_unrouted_verification_one_verdict_per_sentence() -> None:
    u = _unrouted_with(
        draft_sentences=_two_sentences(), verification=_verification(2, supported=False)
    )
    assert [v.sentence_index for v in u.verification.verdicts] == [0, 1]  # type: ignore[union-attr]
    assert UnroutedResult.model_validate_json(u.model_dump_json()) == u


def test_unrouted_verification_none_when_declined() -> None:
    with pytest.raises(ValidationError, match="verification must be None"):
        _unrouted_with(declined_reason="not supported", verification=_verification(1))


def test_unrouted_verification_none_when_no_draft() -> None:
    with pytest.raises(ValidationError, match="verification must be None"):
        _unrouted_with(verification=_verification(1))
    with pytest.raises(ValidationError, match="verification must be None"):
        _unrouted_with(verification=VerificationResult(verdicts=()))


@pytest.mark.parametrize("n_verdicts", [0, 1, 3])
def test_unrouted_verification_count_must_match(n_verdicts: int) -> None:
    with pytest.raises(ValidationError, match="one verdict per draft sentence"):
        _unrouted_with(draft_sentences=_two_sentences(), verification=_verification(n_verdicts))


@pytest.mark.parametrize("indexes", [(1, 0), (0, 0), (1, 2)])
def test_unrouted_verification_indexes_must_be_in_order(indexes: tuple[int, int]) -> None:
    verdicts = tuple(
        SentenceVerdict(sentence_index=i, quotes_match=True, supported=True) for i in indexes
    )
    with pytest.raises(ValidationError, match="sentence_index 0..1"):
        _unrouted_with(
            draft_sentences=_two_sentences(), verification=VerificationResult(verdicts=verdicts)
        )


def test_unrouted_stages_replace_timings_and_token_cost() -> None:
    u = _unrouted()
    assert sum(m.prompt_tokens for m in u.stages) == 100
    for old in ("timings", "token_cost"):
        with pytest.raises(ValidationError):
            _unrouted_with(**{old: ()})
