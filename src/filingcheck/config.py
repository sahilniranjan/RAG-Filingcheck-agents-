"""Settings for FilingCheck, each read from an environment variable with a default.

Use get_settings() in application code (cached for the process) and
load_settings(env) in tests (pure, reads only the mapping you pass).

Defaults here are configuration choices (model names, k values, sizes), not
measured results. Routing thresholds have no default on purpose: they are set only
after evals, see Checkpoint D. Code that routes must call require_routing_thresholds(),
which raises ThresholdsNotSetError while any threshold is None.

Change this file only through contracts-architect, with a stated reason.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import date
from functools import lru_cache

KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    # Local models, run in process with sentence-transformers. No key, no network
    # after the first model download.
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    # OpenAI models, used only with EMBEDDING_PROVIDER=openai.
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
"""Output dimension of known embedding models. EMBEDDING_DIM must match the model,
and sql/schema.sql declares vector(384) to match the default model. Changing the
model to one with a different dimension requires a schema change and a re-ingest."""


KNOWN_EMBEDDING_PROVIDERS: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "local",
    "BAAI/bge-base-en-v1.5": "local",
    "text-embedding-3-small": "openai",
    "text-embedding-3-large": "openai",
    "text-embedding-ada-002": "openai",
}
"""The provider each known embedding model belongs to. load_settings refuses a
known model paired with the wrong EMBEDDING_PROVIDER. It also refuses provider
"local" with any model whose name starts with OPENAI_EMBEDDING_PREFIX, known or not.
Unknown models are otherwise allowed, and then EMBEDDING_DIM is trusted as given."""

OPENAI_EMBEDDING_PREFIX = "text-embedding-"
"""Name prefix of OpenAI embedding models. Never a local sentence-transformers model."""


BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
"""The official BGE v1.5 retrieval instruction for queries (passages get none). It
ends with one space, which separates it from the query text."""

KNOWN_EMBEDDING_QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": BGE_QUERY_INSTRUCTION,
    "BAAI/bge-base-en-v1.5": BGE_QUERY_INSTRUCTION,
    "text-embedding-3-small": "",
    "text-embedding-3-large": "",
    "text-embedding-ada-002": "",
}
"""Default EMBEDDING_QUERY_PREFIX per known model. Unknown models default to "".
A set EMBEDDING_QUERY_PREFIX overrides this (see _query_prefix for normalization)."""


API_KEY_ENV_VARS: dict[str, str | None] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
}
"""The env variable holding each LLM provider's API key. Ollama needs no key."""


RENAMED_ENV_VARS: dict[str, str] = {
    "RERANK_TOP_N": "RERANK_CANDIDATES",
    "VERIFY_MODEL": "VERIFIER_MODEL",
}
"""Old env variable name -> its replacement. load_settings refuses to load if an old
name is set, so a stale .env fails loudly instead of being silently ignored."""


REMOVED_ENV_VARS: dict[str, str] = {
    "GOLD_QUOTE_MAX_TOKENS": (
        "it is now the fixed constant GOLD_QUOTE_MAX_TOKENS in contracts.py, so golden "
        "item validity never depends on env; remove it from your env"
    ),
}
"""Env variables that are no longer settings at all -> why, and what to do. Like
RENAMED_ENV_VARS, load_settings refuses to load if one of these is set."""


def fiscal_year_for(period_of_report: date) -> int:
    """Return the fiscal year for a 10-K. The one definition used everywhere.

    The input is the last day of the reporting period (reportDate in the EDGAR
    submissions JSON), not the filing date.

    Rule:
    - A period ending January 1 through January 7 (inclusive) belongs to the previous
      fiscal year. These are 52/53-week years that run a few days past December 31,
      so a period ending Jan 2 2025 is FY2024.
    - Otherwise the fiscal year is the calendar year in which the period ends. A 10-K
      for the period ending Dec 31 2024 and filed in Feb 2025 is FY2024, a period
      ending June 30 2024 (MSFT style) is FY2024, and a period ending Sep 30 2024
      (V style) is FY2024. A period ending Jan 8 2025 is FY2025.

    edgar-ingest flags any period ending in early January in the coverage report for
    Sahil to check.
    """
    if period_of_report.month == 1 and period_of_report.day <= 7:
        return period_of_report.year - 1
    return period_of_report.year


class ConfigError(ValueError):
    """An environment variable is present but invalid, or settings are inconsistent."""


def missing_api_key_message(provider: str) -> str:
    """Return the one-line message for a missing API key. Pure.

    It names the provider and its exact env variable and says how to fix it. Raises
    ValueError for a provider that needs no key (ollama) or is unknown.
    """
    env_var = API_KEY_ENV_VARS.get(provider)
    if env_var is None:
        raise ValueError(f"provider {provider!r} has no API key env variable")
    return (
        f"LLM provider {provider!r} needs an API key but {env_var} is not set. "
        f"Set {env_var} in .env, or use LLM_MODE=replay once the LLM cache exists."
    )


class MissingAPIKeyError(ConfigError):
    """A live or record LLM call needs an API key that is not set.

    Raised at call time by llm/client.py (see contracts.LLMClient), never by
    load_settings, so LLM_MODE=replay and local-only runs work with no keys at all.
    Entry points (scripts, evals/run_evals.py, the API) catch it and print or return
    only str(error), with no stack trace. provider and env_var are kept as attributes.
    """

    def __init__(self, provider: str) -> None:
        super().__init__(missing_api_key_message(provider))
        self.provider = provider
        self.env_var = API_KEY_ENV_VARS[provider]


def api_key_for(settings: Settings, provider: str) -> str | None:
    """Return the API key for provider, or None for ollama (no key needed).

    Raises MissingAPIKeyError if the provider needs a key and it is unset. Call this
    only right before a live or record call, never in replay mode.
    """
    env_var = API_KEY_ENV_VARS.get(provider)
    if env_var is None:
        if provider in API_KEY_ENV_VARS:
            return None
        raise ConfigError(f"unknown LLM provider {provider!r}")
    key = {"openai": settings.openai_api_key, "anthropic": settings.anthropic_api_key}[provider]
    if key is None:
        raise MissingAPIKeyError(provider)
    return key


class ThresholdsNotSetError(RuntimeError):
    """Routing was attempted while one or more routing thresholds are still None."""


@dataclass(frozen=True)
class RoutingThresholds:
    """Thresholds the router compares against. None means not set yet.

    min_rerank_score: if the top reranked chunk scores below this, route to
    WEAK_EVIDENCE. Set after evals, see Checkpoint D.
    """

    min_rerank_score: float | None = None  # set after evals, see Checkpoint D


@dataclass(frozen=True)
class ReadyRoutingThresholds:
    """Routing thresholds after require_routing_thresholds() confirmed none is None."""

    min_rerank_score: float


@dataclass(frozen=True)
class Settings:
    """Every tunable value in FilingCheck. Field docs name the env variable.

    API key fields are excluded from repr, so printing or logging a Settings object
    never shows a key. Never log the key fields themselves.
    """

    # Secrets and endpoints (no secret defaults). Keys are hidden from repr.
    openai_api_key: str | None = field(repr=False)  # OPENAI_API_KEY
    anthropic_api_key: str | None = field(repr=False)  # ANTHROPIC_API_KEY
    sec_user_agent: str | None  # SEC_USER_AGENT, required by edgar at call time
    database_url: str  # DATABASE_URL

    # EDGAR client
    sec_rate_limit_rps: float  # SEC_RATE_LIMIT_RPS, SEC allows 10, we use 8
    sec_max_retries: int  # SEC_MAX_RETRIES
    sec_backoff_base_seconds: float  # SEC_BACKOFF_BASE_SECONDS
    sec_timeout_seconds: float  # SEC_TIMEOUT_SECONDS
    raw_cache_dir: str  # RAW_CACHE_DIR

    # Chunking
    tokenizer_encoding: str  # TOKENIZER_ENCODING (tiktoken encoding name)
    chunk_min_tokens: int  # CHUNK_MIN_TOKENS
    chunk_max_tokens: int  # CHUNK_MAX_TOKENS
    # CHUNK_OVERLAP_TOKENS, about 15% of a 500 token chunk. Must be at least
    # contracts.GOLD_QUOTE_MAX_TOKENS, so a gold quote always fits whole inside some chunk.
    chunk_overlap_tokens: int

    # Embeddings
    # EMBEDDING_PROVIDER, one of contracts.EMBEDDING_PROVIDERS. "local" runs a
    # sentence-transformers model in process and needs no key.
    embedding_provider: str
    embedding_model: str  # EMBEDDING_MODEL, must belong to embedding_provider
    embedding_dim: int  # EMBEDDING_DIM, must match embedding_model and sql/schema.sql
    # EMBEDDING_BATCH_SIZE, texts per embedding call. 32 suits a local model on a
    # laptop CPU; raise it (for example to 100) for the OpenAI API.
    embedding_batch_size: int
    # EMBEDDING_QUERY_PREFIX, prepended to queries only (contracts.embedding_input).
    # Defaults per model from KNOWN_EMBEDDING_QUERY_PREFIXES, "" for unknown models.
    embedding_query_prefix: str

    # Retrieval
    bm25_k: int  # BM25_K, candidates from full-text search
    vector_k: int  # VECTOR_K, candidates from vector search
    rrf_k: int  # RRF_K, the Reciprocal Rank Fusion constant
    rerank_model: str  # RERANK_MODEL, any sentence-transformers CrossEncoder name
    rerank_candidates: int  # RERANK_CANDIDATES, fused candidates sent to the reranker
    rerank_top_k: int  # RERANK_TOP_K, chunks kept after rerank; what evals measure (Recall@10)
    generation_k: int  # GENERATION_K, top reranked chunks put into the answer prompt
    default_retrieval_mode: str  # RETRIEVAL_MODE

    # LLMs. Every provider field is one of contracts.LLM_PROVIDERS. All calls go
    # through contracts.LLMClient, never a provider SDK directly.
    generation_provider: str  # LLM_PROVIDER, used for answer generation
    generation_model: str  # GENERATION_MODEL
    verifier_provider: str  # VERIFIER_PROVIDER
    verifier_model: str  # VERIFIER_MODEL
    # The judge should be a different model family from the generator to reduce
    # self-preference bias, confirmed by the hand calibration at Checkpoint D.
    judge_provider: str  # JUDGE_PROVIDER (evals only)
    judge_model: str  # JUDGE_MODEL (evals only)
    llm_temperature: float  # LLM_TEMPERATURE
    llm_timeout_seconds: float  # LLM_TIMEOUT_SECONDS
    ollama_base_url: str  # OLLAMA_BASE_URL, used when any provider is "ollama"
    # LLM_MODE, one of contracts.LLM_MODES: "live" calls the provider, "record" calls
    # it and writes the cache, "replay" reads only the cache and never uses the network.
    llm_mode: str

    # Routing
    routing: RoutingThresholds  # ROUTE_MIN_RERANK_SCORE


def _get(env: Mapping[str, str], name: str) -> str | None:
    """Return the env value, treating an empty or whitespace-only string as unset."""
    value = env.get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _str(env: Mapping[str, str], name: str, default: str) -> str:
    return _get(env, name) or default


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _get(env, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _get(env, name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _optional_float(env: Mapping[str, str], name: str) -> float | None:
    raw = _get(env, name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number or unset, got {raw!r}") from exc


def _query_prefix(env: Mapping[str, str], embedding_model: str) -> str:
    """Return EMBEDDING_QUERY_PREFIX, or the model default when it is unset or blank.

    Many .env loaders strip trailing spaces, so a set value is normalized: surrounding
    whitespace is stripped and exactly one space is appended. "query:" and "query: "
    both become "query: ". The prefix is then joined to the query with plain
    concatenation. An unset or whitespace-only value means use the model default.
    """
    raw = _get(env, "EMBEDDING_QUERY_PREFIX")
    if raw is None:
        return KNOWN_EMBEDDING_QUERY_PREFIXES.get(embedding_model, "")
    return raw + " "


def _refuse_renamed(env: Mapping[str, str]) -> None:
    """Raise ConfigError if any renamed or removed env variable is set to a non-empty value."""
    for old, new in RENAMED_ENV_VARS.items():
        if _get(env, old) is not None:
            raise ConfigError(f"{old} is no longer read; rename it to {new} in your env")
    for old, why in REMOVED_ENV_VARS.items():
        if _get(env, old) is not None:
            raise ConfigError(f"{old} is no longer read: {why}")


def _check_embedding_pair(provider: str, model: str) -> None:
    """Raise ConfigError if the embedding model obviously belongs to another provider.

    A known model must match its entry in KNOWN_EMBEDDING_PROVIDERS. Provider "local"
    also refuses any model named like an OpenAI one (OPENAI_EMBEDDING_PREFIX). Other
    unknown models pass, since a local sentence-transformers model can have any name.
    """
    owner = KNOWN_EMBEDDING_PROVIDERS.get(model)
    if owner is None and provider == "local" and model.startswith(OPENAI_EMBEDDING_PREFIX):
        owner = "openai"
    if owner is not None and owner != provider:
        raise ConfigError(
            f"EMBEDDING_MODEL {model!r} is a {owner} model but EMBEDDING_PROVIDER is "
            f"{provider!r}; set EMBEDDING_PROVIDER={owner} or pick a {provider} model"
        )


def _validate(settings: Settings) -> None:
    """Raise ConfigError if settings are internally inconsistent."""
    # Imported here, not at module level, because contracts.py imports config.
    from filingcheck.contracts import (
        EMBEDDING_PROVIDERS,
        GOLD_QUOTE_MAX_TOKENS,
        LLM_MODES,
        LLM_PROVIDERS,
        RETRIEVAL_MODES,
    )

    if settings.embedding_provider not in EMBEDDING_PROVIDERS:
        raise ConfigError(
            f"unknown EMBEDDING_PROVIDER {settings.embedding_provider!r}, "
            f"expected one of {', '.join(EMBEDDING_PROVIDERS)}"
        )
    _check_embedding_pair(settings.embedding_provider, settings.embedding_model)
    for env_name, provider in (
        ("LLM_PROVIDER", settings.generation_provider),
        ("VERIFIER_PROVIDER", settings.verifier_provider),
        ("JUDGE_PROVIDER", settings.judge_provider),
    ):
        if provider not in LLM_PROVIDERS:
            raise ConfigError(
                f"unknown {env_name} {provider!r}, expected one of {', '.join(LLM_PROVIDERS)}"
            )
    if settings.llm_mode not in LLM_MODES:
        raise ConfigError(
            f"unknown LLM_MODE {settings.llm_mode!r}, expected one of {', '.join(LLM_MODES)}"
        )
    if not settings.ollama_base_url.startswith(("http://", "https://")):
        raise ConfigError(
            f"OLLAMA_BASE_URL must start with http:// or https://, got {settings.ollama_base_url!r}"
        )
    if settings.embedding_batch_size < 1:
        raise ConfigError("EMBEDDING_BATCH_SIZE must be at least 1")

    known_dim = KNOWN_EMBEDDING_DIMS.get(settings.embedding_model)
    if known_dim is not None and known_dim != settings.embedding_dim:
        raise ConfigError(
            f"EMBEDDING_DIM {settings.embedding_dim} does not match "
            f"{settings.embedding_model} (dimension {known_dim})"
        )
    if not 0 < settings.chunk_min_tokens <= settings.chunk_max_tokens:
        raise ConfigError("need 0 < CHUNK_MIN_TOKENS <= CHUNK_MAX_TOKENS")
    if not 0 <= settings.chunk_overlap_tokens < settings.chunk_min_tokens:
        raise ConfigError("need 0 <= CHUNK_OVERLAP_TOKENS < CHUNK_MIN_TOKENS")
    if settings.chunk_overlap_tokens < GOLD_QUOTE_MAX_TOKENS:
        raise ConfigError(
            f"CHUNK_OVERLAP_TOKENS {settings.chunk_overlap_tokens} is below the gold quote "
            f"cap of {GOLD_QUOTE_MAX_TOKENS} tokens (contracts.GOLD_QUOTE_MAX_TOKENS). A gold "
            "quote must fit whole inside a chunk overlap, so no chunk boundary can cut it; "
            f"set CHUNK_OVERLAP_TOKENS to at least {GOLD_QUOTE_MAX_TOKENS}"
        )
    if settings.generation_k < 1:
        raise ConfigError("GENERATION_K must be at least 1")
    if settings.generation_k > settings.rerank_top_k:
        raise ConfigError("GENERATION_K must not exceed RERANK_TOP_K")
    if settings.rerank_top_k > settings.rerank_candidates:
        raise ConfigError("RERANK_TOP_K must not exceed RERANK_CANDIDATES")
    if settings.sec_rate_limit_rps <= 0 or settings.sec_rate_limit_rps >= 10:
        raise ConfigError("SEC_RATE_LIMIT_RPS must be above 0 and below 10 (SEC limit)")
    if settings.default_retrieval_mode not in RETRIEVAL_MODES:
        raise ConfigError(f"unknown RETRIEVAL_MODE {settings.default_retrieval_mode!r}")


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from env (defaults to os.environ). Pure apart from reading env.

    Raises ConfigError if a value cannot be parsed, settings are inconsistent, or an
    old renamed or removed variable (see RENAMED_ENV_VARS and REMOVED_ENV_VARS) is set.
    """
    e: Mapping[str, str] = os.environ if env is None else env
    _refuse_renamed(e)
    embedding_model = _str(e, "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    settings = Settings(
        openai_api_key=_get(e, "OPENAI_API_KEY"),
        anthropic_api_key=_get(e, "ANTHROPIC_API_KEY"),
        sec_user_agent=_get(e, "SEC_USER_AGENT"),
        database_url=_str(
            e, "DATABASE_URL", "postgresql://filingcheck:filingcheck@localhost:5432/filingcheck"
        ),
        sec_rate_limit_rps=_float(e, "SEC_RATE_LIMIT_RPS", 8.0),
        sec_max_retries=_int(e, "SEC_MAX_RETRIES", 5),
        sec_backoff_base_seconds=_float(e, "SEC_BACKOFF_BASE_SECONDS", 1.0),
        sec_timeout_seconds=_float(e, "SEC_TIMEOUT_SECONDS", 30.0),
        raw_cache_dir=_str(e, "RAW_CACHE_DIR", "data/raw"),
        tokenizer_encoding=_str(e, "TOKENIZER_ENCODING", "cl100k_base"),
        chunk_min_tokens=_int(e, "CHUNK_MIN_TOKENS", 400),
        chunk_max_tokens=_int(e, "CHUNK_MAX_TOKENS", 600),
        chunk_overlap_tokens=_int(e, "CHUNK_OVERLAP_TOKENS", 75),
        embedding_provider=_str(e, "EMBEDDING_PROVIDER", "local"),
        embedding_model=embedding_model,
        embedding_dim=_int(e, "EMBEDDING_DIM", 384),
        embedding_batch_size=_int(e, "EMBEDDING_BATCH_SIZE", 32),
        embedding_query_prefix=_query_prefix(e, embedding_model),
        bm25_k=_int(e, "BM25_K", 50),
        vector_k=_int(e, "VECTOR_K", 50),
        rrf_k=_int(e, "RRF_K", 60),
        rerank_model=_str(e, "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        rerank_candidates=_int(e, "RERANK_CANDIDATES", 30),
        rerank_top_k=_int(e, "RERANK_TOP_K", 10),
        generation_k=_int(e, "GENERATION_K", 6),
        default_retrieval_mode=_str(e, "RETRIEVAL_MODE", "hybrid_rerank"),
        generation_provider=_str(e, "LLM_PROVIDER", "openai"),
        generation_model=_str(e, "GENERATION_MODEL", "gpt-4o-mini"),
        verifier_provider=_str(e, "VERIFIER_PROVIDER", "openai"),
        verifier_model=_str(e, "VERIFIER_MODEL", "gpt-4o-mini"),
        # Judge: prefer a different model family from the generator to reduce
        # self-preference bias, confirmed by the hand calibration at Checkpoint D.
        judge_provider=_str(e, "JUDGE_PROVIDER", "openai"),
        judge_model=_str(e, "JUDGE_MODEL", "gpt-4o"),
        llm_temperature=_float(e, "LLM_TEMPERATURE", 0.0),
        llm_timeout_seconds=_float(e, "LLM_TIMEOUT_SECONDS", 60.0),
        ollama_base_url=_str(e, "OLLAMA_BASE_URL", "http://localhost:11434"),
        llm_mode=_str(e, "LLM_MODE", "live"),
        routing=RoutingThresholds(
            # set after evals, see Checkpoint D
            min_rerank_score=_optional_float(e, "ROUTE_MIN_RERANK_SCORE"),
        ),
    )
    _validate(settings)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide Settings from os.environ, loaded once and cached."""
    return load_settings()


def require_routing_thresholds(settings: Settings | None = None) -> ReadyRoutingThresholds:
    """Return routing thresholds, or raise ThresholdsNotSetError if any is None.

    Every routing code path must call this before comparing against a threshold.
    """
    routing = (settings or get_settings()).routing
    missing = [f.name for f in fields(routing) if getattr(routing, f.name) is None]
    if missing:
        raise ThresholdsNotSetError(
            "Routing thresholds are not set: "
            + ", ".join(missing)
            + ". They are set only after evals (see Checkpoint D), via env "
            "ROUTE_MIN_RERANK_SCORE. Refusing to route."
        )
    return ReadyRoutingThresholds(min_rerank_score=float(routing.min_rerank_score))  # type: ignore[arg-type]


__all__ = [
    "KNOWN_EMBEDDING_DIMS",
    "API_KEY_ENV_VARS",
    "BGE_QUERY_INSTRUCTION",
    "KNOWN_EMBEDDING_PROVIDERS",
    "KNOWN_EMBEDDING_QUERY_PREFIXES",
    "MissingAPIKeyError",
    "api_key_for",
    "missing_api_key_message",
    "OPENAI_EMBEDDING_PREFIX",
    "REMOVED_ENV_VARS",
    "RENAMED_ENV_VARS",
    "ConfigError",
    "ReadyRoutingThresholds",
    "RoutingThresholds",
    "Settings",
    "ThresholdsNotSetError",
    "fiscal_year_for",
    "get_settings",
    "load_settings",
    "require_routing_thresholds",
]
