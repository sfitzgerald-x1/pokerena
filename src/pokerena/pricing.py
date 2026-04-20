from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional

from .config import AgentDefinition, AgentPricingConfig


USD_QUANTIZE = Decimal("0.000001")
MILLION_TOKENS = Decimal("1000000")


@dataclass(frozen=True)
class TokenPricing:
    input_usd_per_million_tokens: Decimal
    output_usd_per_million_tokens: Decimal
    cache_read_input_usd_per_million_tokens: Decimal
    cache_creation_input_usd_per_million_tokens: Decimal


@dataclass(frozen=True)
class ResolvedPricing:
    model: str
    pricing_source: str
    rates: TokenPricing


DEFAULT_MODEL_PRICING: Dict[str, TokenPricing] = {
    # Anthropic Claude API pricing, accessed April 19, 2026:
    # https://platform.claude.com/docs/es/about-claude/pricing
    "claude-opus-4-7": TokenPricing(
        input_usd_per_million_tokens=Decimal("5"),
        output_usd_per_million_tokens=Decimal("25"),
        cache_read_input_usd_per_million_tokens=Decimal("0.50"),
        cache_creation_input_usd_per_million_tokens=Decimal("6.25"),
    ),
    "claude-opus-4-6": TokenPricing(
        input_usd_per_million_tokens=Decimal("5"),
        output_usd_per_million_tokens=Decimal("25"),
        cache_read_input_usd_per_million_tokens=Decimal("0.50"),
        cache_creation_input_usd_per_million_tokens=Decimal("6.25"),
    ),
    "claude-opus-4-5": TokenPricing(
        input_usd_per_million_tokens=Decimal("5"),
        output_usd_per_million_tokens=Decimal("25"),
        cache_read_input_usd_per_million_tokens=Decimal("0.50"),
        cache_creation_input_usd_per_million_tokens=Decimal("6.25"),
    ),
    "claude-opus-4-1": TokenPricing(
        input_usd_per_million_tokens=Decimal("15"),
        output_usd_per_million_tokens=Decimal("75"),
        cache_read_input_usd_per_million_tokens=Decimal("1.50"),
        cache_creation_input_usd_per_million_tokens=Decimal("18.75"),
    ),
    "claude-opus-4": TokenPricing(
        input_usd_per_million_tokens=Decimal("15"),
        output_usd_per_million_tokens=Decimal("75"),
        cache_read_input_usd_per_million_tokens=Decimal("1.50"),
        cache_creation_input_usd_per_million_tokens=Decimal("18.75"),
    ),
    "claude-sonnet-4-6": TokenPricing(
        input_usd_per_million_tokens=Decimal("3"),
        output_usd_per_million_tokens=Decimal("15"),
        cache_read_input_usd_per_million_tokens=Decimal("0.30"),
        cache_creation_input_usd_per_million_tokens=Decimal("3.75"),
    ),
    "claude-sonnet-4-5": TokenPricing(
        input_usd_per_million_tokens=Decimal("3"),
        output_usd_per_million_tokens=Decimal("15"),
        cache_read_input_usd_per_million_tokens=Decimal("0.30"),
        cache_creation_input_usd_per_million_tokens=Decimal("3.75"),
    ),
    "claude-sonnet-4": TokenPricing(
        input_usd_per_million_tokens=Decimal("3"),
        output_usd_per_million_tokens=Decimal("15"),
        cache_read_input_usd_per_million_tokens=Decimal("0.30"),
        cache_creation_input_usd_per_million_tokens=Decimal("3.75"),
    ),
    "claude-3-7-sonnet": TokenPricing(
        input_usd_per_million_tokens=Decimal("3"),
        output_usd_per_million_tokens=Decimal("15"),
        cache_read_input_usd_per_million_tokens=Decimal("0.30"),
        cache_creation_input_usd_per_million_tokens=Decimal("3.75"),
    ),
    "claude-haiku-4-5": TokenPricing(
        input_usd_per_million_tokens=Decimal("1"),
        output_usd_per_million_tokens=Decimal("5"),
        cache_read_input_usd_per_million_tokens=Decimal("0.10"),
        cache_creation_input_usd_per_million_tokens=Decimal("1.25"),
    ),
    "claude-haiku-3-5": TokenPricing(
        input_usd_per_million_tokens=Decimal("0.80"),
        output_usd_per_million_tokens=Decimal("4"),
        cache_read_input_usd_per_million_tokens=Decimal("0.08"),
        cache_creation_input_usd_per_million_tokens=Decimal("1"),
    ),
}


def resolve_agent_model(agent: AgentDefinition) -> Optional[str]:
    if agent.pricing.model:
        return _normalize_model_name(agent.pricing.model)
    args = list(agent.launch.args)
    for index, arg in enumerate(args):
        if arg == "--model" and index + 1 < len(args):
            return _normalize_model_name(args[index + 1])
        if isinstance(arg, str) and arg.startswith("--model="):
            return _normalize_model_name(arg.partition("=")[2])
    return None


def pricing_snapshot_for_agent(agent: AgentDefinition) -> Optional[Dict[str, Any]]:
    resolved = resolve_pricing_for_agent(agent)
    if resolved is None:
        return None
    return {
        "model": resolved.model,
        "pricing_source": resolved.pricing_source,
        "input_usd_per_million_tokens": _decimal_to_string(
            resolved.rates.input_usd_per_million_tokens
        ),
        "output_usd_per_million_tokens": _decimal_to_string(
            resolved.rates.output_usd_per_million_tokens
        ),
        "cache_read_input_usd_per_million_tokens": _decimal_to_string(
            resolved.rates.cache_read_input_usd_per_million_tokens
        ),
        "cache_creation_input_usd_per_million_tokens": _decimal_to_string(
            resolved.rates.cache_creation_input_usd_per_million_tokens
        ),
    }


def resolve_pricing_for_agent(agent: AgentDefinition) -> Optional[ResolvedPricing]:
    model = resolve_agent_model(agent)
    if model is None:
        return None
    return resolve_pricing_snapshot(model=model, pricing_override=agent.pricing)


def resolve_pricing_snapshot(
    *,
    model: Optional[str],
    pricing_override: Optional[AgentPricingConfig] = None,
    pricing_snapshot: Optional[Dict[str, Any]] = None,
) -> Optional[ResolvedPricing]:
    normalized_model = _normalize_model_name(model)
    if pricing_snapshot:
        return _resolved_from_snapshot(pricing_snapshot, fallback_model=normalized_model)
    if normalized_model is None:
        return None

    default_rates = default_pricing_for_model(normalized_model)
    override_rates = pricing_override or AgentPricingConfig()
    merged = TokenPricing(
        input_usd_per_million_tokens=override_rates.input_usd_per_million_tokens
        or (default_rates.input_usd_per_million_tokens if default_rates else None),
        output_usd_per_million_tokens=override_rates.output_usd_per_million_tokens
        or (default_rates.output_usd_per_million_tokens if default_rates else None),
        cache_read_input_usd_per_million_tokens=override_rates.cache_read_input_usd_per_million_tokens
        or (default_rates.cache_read_input_usd_per_million_tokens if default_rates else None),
        cache_creation_input_usd_per_million_tokens=override_rates.cache_creation_input_usd_per_million_tokens
        or (default_rates.cache_creation_input_usd_per_million_tokens if default_rates else None),
    )
    if not all(
        (
            merged.input_usd_per_million_tokens,
            merged.output_usd_per_million_tokens,
            merged.cache_read_input_usd_per_million_tokens,
            merged.cache_creation_input_usd_per_million_tokens,
        )
    ):
        return None
    pricing_source = "agent-override" if _has_override_values(override_rates) else "default"
    return ResolvedPricing(model=normalized_model, pricing_source=pricing_source, rates=merged)


def default_pricing_for_model(model: str) -> Optional[TokenPricing]:
    normalized = _normalize_model_name(model)
    if normalized is None:
        return None
    if normalized in DEFAULT_MODEL_PRICING:
        return DEFAULT_MODEL_PRICING[normalized]
    for candidate, pricing in DEFAULT_MODEL_PRICING.items():
        if normalized.startswith(f"{candidate}-"):
            return pricing
    return None


def estimate_usage_cost(
    usage: object,
    *,
    model: Optional[str],
    pricing_snapshot: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(usage, dict):
        return None
    resolved = resolve_pricing_snapshot(model=model, pricing_snapshot=pricing_snapshot)
    if resolved is None:
        return None

    token_counts = {
        "input_tokens": _usage_int(usage.get("input_tokens")),
        "output_tokens": _usage_int(usage.get("output_tokens")),
        "cache_read_input_tokens": _usage_int(usage.get("cache_read_input_tokens")),
        "cache_creation_input_tokens": _usage_int(usage.get("cache_creation_input_tokens")),
    }
    if not any(value is not None for value in token_counts.values()):
        return None

    breakdown = {
        "input_usd": _cost_for_tokens(
            token_counts["input_tokens"], resolved.rates.input_usd_per_million_tokens
        ),
        "output_usd": _cost_for_tokens(
            token_counts["output_tokens"], resolved.rates.output_usd_per_million_tokens
        ),
        "cache_read_input_usd": _cost_for_tokens(
            token_counts["cache_read_input_tokens"],
            resolved.rates.cache_read_input_usd_per_million_tokens,
        ),
        "cache_creation_input_usd": _cost_for_tokens(
            token_counts["cache_creation_input_tokens"],
            resolved.rates.cache_creation_input_usd_per_million_tokens,
        ),
    }
    total = sum(value for value in breakdown.values() if value is not None)
    payload: Dict[str, Any] = {
        "currency": "USD",
        "model": resolved.model,
        "pricing_source": resolved.pricing_source,
        "total_usd": _decimal_to_string(total),
    }
    for key, value in breakdown.items():
        if value is not None:
            payload[key] = _decimal_to_string(value)
    return payload


def _resolved_from_snapshot(
    snapshot: Dict[str, Any],
    *,
    fallback_model: Optional[str],
) -> Optional[ResolvedPricing]:
    model_value = snapshot.get("model")
    pricing_source = snapshot.get("pricing_source")
    if not isinstance(pricing_source, str) or pricing_source not in {"default", "agent-override"}:
        pricing_source = "default"
    model = _normalize_model_name(model_value if isinstance(model_value, str) else fallback_model)
    if model is None:
        return None
    rates = TokenPricing(
        input_usd_per_million_tokens=_decimal_from_snapshot(
            snapshot.get("input_usd_per_million_tokens")
        ),
        output_usd_per_million_tokens=_decimal_from_snapshot(
            snapshot.get("output_usd_per_million_tokens")
        ),
        cache_read_input_usd_per_million_tokens=_decimal_from_snapshot(
            snapshot.get("cache_read_input_usd_per_million_tokens")
        ),
        cache_creation_input_usd_per_million_tokens=_decimal_from_snapshot(
            snapshot.get("cache_creation_input_usd_per_million_tokens")
        ),
    )
    if not all(
        (
            rates.input_usd_per_million_tokens,
            rates.output_usd_per_million_tokens,
            rates.cache_read_input_usd_per_million_tokens,
            rates.cache_creation_input_usd_per_million_tokens,
        )
    ):
        return None
    return ResolvedPricing(model=model, pricing_source=pricing_source, rates=rates)


def _cost_for_tokens(token_count: Optional[int], rate: Decimal) -> Optional[Decimal]:
    if token_count is None:
        return None
    return (Decimal(token_count) * rate / MILLION_TOKENS).quantize(
        USD_QUANTIZE, rounding=ROUND_HALF_UP
    )


def _decimal_from_snapshot(value: object) -> Optional[Decimal]:
    if isinstance(value, str) and value.strip():
        return Decimal(value)
    return None


def _decimal_to_string(value: Decimal) -> str:
    return format(value.quantize(USD_QUANTIZE, rounding=ROUND_HALF_UP), "f")


def _normalize_model_name(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _has_override_values(pricing: AgentPricingConfig) -> bool:
    return any(
        value is not None
        for value in (
            pricing.input_usd_per_million_tokens,
            pricing.output_usd_per_million_tokens,
            pricing.cache_read_input_usd_per_million_tokens,
            pricing.cache_creation_input_usd_per_million_tokens,
        )
    )


def _usage_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
