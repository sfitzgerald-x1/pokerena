"""Deterministic Gen 1 randbat baseline that always chooses max damage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

from .agent import AgentDecision, choose_first_legal
from .calc import DEFAULT_CALC_TIMEOUT_SECONDS, detect_project_root
from .config import ConfigError
from .custom_bot import (
    CustomBotPlan,
    MoveCandidate,
    ScoredAction,
    _active_opponent_from_public_history,
    _active_side_pokemon,
    _context_side,
    _damage_results_by_choice,
    _float_value,
    _is_damaging_metadata,
    _int_value,
    _is_supported_gen1_randbat,
    _move_candidates,
    _public_lines,
    _scored_action,
    _single_active_request,
)


@dataclass(frozen=True)
class MaxDamageCandidate:
    action: ScoredAction
    move: MoveCandidate
    min_percent: float
    max_percent: float


def decide_max_damage_bot_from_files(
    *,
    context_path: Optional[str],
    capture_path: Optional[str],
    project_root: Optional[Path] = None,
    calc_timeout_seconds: int = DEFAULT_CALC_TIMEOUT_SECONDS,
) -> AgentDecision:
    resolved_context_path = Path(
        context_path or os.environ.get("POKERENA_TURN_CONTEXT_PATH") or ""
    )
    if not resolved_context_path.exists():
        raise ConfigError("Max damage bot requires --context or POKERENA_TURN_CONTEXT_PATH.")
    context_payload = _read_json_object(resolved_context_path, "turn context")

    capture_value = capture_path or os.environ.get("POKERENA_BATTLE_CAPTURE_PATH")
    resolved_capture_path = Path(capture_value) if capture_value else None
    capture_payload = (
        _read_json_object(resolved_capture_path, "battle capture")
        if resolved_capture_path is not None and resolved_capture_path.exists()
        else None
    )
    plan = build_max_damage_bot_plan(
        context_payload,
        capture_payload=capture_payload,
        project_root=project_root,
        calc_timeout_seconds=calc_timeout_seconds,
    )
    for warning in plan.warnings or []:
        print(warning, file=sys.stderr)
    return AgentDecision(
        schema_version="pokerena.decision.v1",
        decision=plan.decision,
        notes=plan.notes,
        raw_output="",
    )


def emit_max_damage_bot_decision(decision: AgentDecision) -> None:
    print(json.dumps(asdict(decision), separators=(",", ":")))


def build_max_damage_bot_plan(
    context: Dict[str, Any],
    *,
    capture_payload: Optional[Dict[str, Any]] = None,
    project_root: Optional[Path] = None,
    calc_timeout_seconds: int = DEFAULT_CALC_TIMEOUT_SECONDS,
) -> CustomBotPlan:
    request = context.get("request") if isinstance(context.get("request"), dict) else None
    if not request:
        return _fallback_plan(request, "no active request")

    if str(context.get("request_kind") or "") != "move" or not _is_supported_gen1_randbat(context):
        return _fallback_plan(request, "unsupported format or request kind")

    active_request = _single_active_request(context)
    own_active = _active_side_pokemon(_context_side(context))
    public_lines = _public_lines(context, capture_payload)
    opponent = _active_opponent_from_public_history(context, public_lines)
    if active_request is None or own_active is None or opponent is None:
        return _fallback_plan(request, "missing active Pokemon or opponent state")

    root = project_root or detect_project_root()
    move_candidates = _move_candidates(
        active_request=active_request,
        project_root=root,
        calc_timeout_seconds=calc_timeout_seconds,
    )
    damaging_candidates = [
        candidate for candidate in move_candidates if _is_damaging_metadata(candidate.metadata)
    ]
    if not damaging_candidates:
        return _fallback_plan(request, "no enabled damaging moves")

    damage_results = _damage_results_by_choice(
        candidates=damaging_candidates,
        attacker=own_active,
        defender=opponent,
        project_root=root,
        calc_timeout_seconds=calc_timeout_seconds,
    )
    scored = _rank_damage_candidates(damaging_candidates, damage_results.results)
    if not scored:
        return _fallback_plan(
            request,
            "damage calc returned no usable damaging results",
            warnings=[damage_results.warning] if damage_results.warning else None,
        )

    decision = scored[0].action.choice
    actions = [candidate.action for candidate in scored]
    notes = _notes(decision, scored, warnings=[damage_results.warning] if damage_results.warning else None)
    return CustomBotPlan(
        decision=decision,
        notes=notes,
        actions=actions,
        warnings=[damage_results.warning] if damage_results.warning else [],
    )


def _rank_damage_candidates(
    candidates: Sequence[MoveCandidate],
    damage_results: Dict[str, Dict[str, Any]],
) -> List[MaxDamageCandidate]:
    ranked: List[MaxDamageCandidate] = []
    for candidate in candidates:
        result = damage_results.get(candidate.choice)
        range_percent = result.get("range_percent") if isinstance(result, dict) else None
        if isinstance(range_percent, dict):
            min_percent = _float_value(range_percent.get("min")) or 0.0
            max_percent = _float_value(range_percent.get("max")) or 0.0
            mean_percent = (min_percent + max_percent) / 2.0
            reason = f"mean damage {mean_percent:.1f}% ({min_percent:.1f}-{max_percent:.1f}%)"
        else:
            base_power = _int_value(candidate.metadata.get("base_power"), 0)
            if base_power <= 0:
                continue
            # Keep uncalculated damaging moves ordered by approximate raw power
            # instead of degenerating into first-legal behavior.
            mean_percent = base_power * 0.25
            min_percent = mean_percent
            max_percent = mean_percent
            reason = f"estimated damage {mean_percent:.1f}% from base power {base_power}"
        ranked.append(
            MaxDamageCandidate(
                action=_scored_action(
                    candidate.choice,
                    candidate.name,
                    mean_percent,
                    reason,
                ),
                move=candidate,
                min_percent=min_percent,
                max_percent=max_percent,
            )
        )
    return sorted(ranked, key=lambda item: (-item.action.score, item.move.index))


def _fallback_plan(
    request: Optional[Dict[str, Any]],
    reason: str,
    *,
    warnings: Optional[Sequence[str]] = None,
) -> CustomBotPlan:
    decision = choose_first_legal(request)
    warning_text = " ".join(warnings or [])
    notes = f"{warning_text} max-damage-bot fallback: {reason}; selected {decision}.".strip()
    return CustomBotPlan(
        decision=decision,
        notes=notes,
        actions=[],
        fallback_reason=reason,
        warnings=list(warnings or []),
    )


def _notes(
    decision: str,
    candidates: Sequence[MaxDamageCandidate],
    *,
    warnings: Optional[Sequence[str]] = None,
) -> str:
    ranked = candidates[:6]
    parts = [
        f"{candidate.action.choice} {candidate.action.label}: {candidate.action.reason}"
        for candidate in ranked
    ]
    warning_text = " ".join(warnings or [])
    score_text = f"max-damage-bot gen1randombattle; selected {decision}. " + "; ".join(parts)
    return f"{warning_text} {score_text}".strip()


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"Failed to read {label} JSON from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigError(f"{label.capitalize()} at {path} must contain a JSON object.")
    return payload
