from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence

from .agent import AgentDecision, parse_decision_output
from .calc import DEFAULT_CALC_TIMEOUT_SECONDS, detect_project_root
from .config import ConfigError
from .custom_bot import (
    MAJOR_STATUSES,
    SLEEP_MOVES,
    _active_opponent_from_public_history,
    _active_side_pokemon,
    _context_side,
    _normalize_move_id,
    _public_lines,
    build_custom_bot_plan,
    emit_custom_bot_decision,
)


DEFAULT_CLAUDE_OVERRIDE_MODEL = "claude-opus-4-7"
DEFAULT_CLAUDE_OVERRIDE_TIMEOUT_SECONDS = 60
REDUNDANT_MAJOR_STATUS_MOVE_IDS = SLEEP_MOVES | {
    "glare",
    "poisonpowder",
    "stunspore",
    "thunderwave",
    "toxic",
}


def decide_custom_bot_claude_from_files(
    *,
    context_path: Optional[str],
    capture_path: Optional[str],
    seed: Optional[str],
    project_root: Optional[Path] = None,
    calc_timeout_seconds: int = DEFAULT_CALC_TIMEOUT_SECONDS,
    claude_command: str = "claude",
    model: str = DEFAULT_CLAUDE_OVERRIDE_MODEL,
    claude_timeout_seconds: int = DEFAULT_CLAUDE_OVERRIDE_TIMEOUT_SECONDS,
) -> AgentDecision:
    resolved_context_path = Path(
        context_path or os.environ.get("POKERENA_TURN_CONTEXT_PATH") or ""
    )
    if not resolved_context_path.exists():
        raise ConfigError("Claude override bot requires --context or POKERENA_TURN_CONTEXT_PATH.")
    context_payload = _read_json_object(resolved_context_path, "turn context")

    capture_value = capture_path or os.environ.get("POKERENA_BATTLE_CAPTURE_PATH")
    resolved_capture_path = Path(capture_value) if capture_value else None
    capture_payload = (
        _read_json_object(resolved_capture_path, "battle capture")
        if resolved_capture_path is not None and resolved_capture_path.exists()
        else None
    )
    rng = random.Random(seed) if seed is not None else random.Random()
    root = project_root or detect_project_root()
    plan = build_custom_bot_plan(
        context_payload,
        capture_payload=capture_payload,
        project_root=root,
        rng=rng,
        calc_timeout_seconds=calc_timeout_seconds,
    )
    for warning in plan.warnings or []:
        print(warning, file=sys.stderr)

    baseline = AgentDecision(
        schema_version="pokerena.decision.v1",
        decision=plan.decision,
        notes=plan.notes,
        raw_output="",
    )
    prompt = build_claude_override_prompt(
        context_payload=context_payload,
        capture_payload=capture_payload,
        baseline_decision=baseline,
        plan_actions=[asdict(action) for action in plan.actions],
        model=model,
    )
    legal_choices = _legal_choices(context_payload)
    legal_choices.add(plan.decision)

    try:
        completed = subprocess.run(
            [claude_command, "-p", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=claude_timeout_seconds,
            check=True,
            cwd=root,
        )
        claude_decision = parse_decision_output(completed.stdout, "pokerena.decision.v1")
    except (OSError, subprocess.SubprocessError, ConfigError) as error:
        return _baseline_with_note(
            baseline,
            f"claude override unavailable; used custom-bot baseline ({error}).",
        )

    if claude_decision.decision not in legal_choices:
        return _baseline_with_note(
            baseline,
            f"claude override ignored illegal decision {claude_decision.decision!r}.",
        )
    tactical_rejection = _override_rejection_reason(context_payload, capture_payload, claude_decision.decision)
    if tactical_rejection is not None:
        return _baseline_with_note(
            baseline,
            f"claude override ignored {claude_decision.decision!r}: {tactical_rejection}.",
        )
    if claude_decision.decision == baseline.decision:
        return AgentDecision(
            schema_version="pokerena.decision.v1",
            decision=baseline.decision,
            notes=_join_notes(
                baseline.notes,
                f"claude override reviewed baseline with {model}; kept baseline. {claude_decision.notes}",
            ),
            raw_output=claude_decision.raw_output,
        )
    return AgentDecision(
        schema_version="pokerena.decision.v1",
        decision=claude_decision.decision,
        notes=_join_notes(
            baseline.notes,
            (
                f"claude override changed {baseline.decision} -> {claude_decision.decision} "
                f"with {model}. {claude_decision.notes}"
            ),
        ),
        raw_output=claude_decision.raw_output,
    )


def emit_custom_bot_claude_decision(decision: AgentDecision) -> None:
    emit_custom_bot_decision(decision)


def build_claude_override_prompt(
    *,
    context_payload: Dict[str, Any],
    capture_payload: Optional[Dict[str, Any]] = None,
    baseline_decision: AgentDecision,
    plan_actions: Sequence[Dict[str, Any]],
    model: str,
) -> str:
    legal_choices = sorted(_legal_choices(context_payload))
    payload = {
        "baseline_decision": asdict(baseline_decision),
        "battle_summary": _battle_summary(context_payload, capture_payload),
        "ranked_custom_bot_actions": list(plan_actions),
        "legal_choices": legal_choices,
        "turn_context": context_payload,
    }
    return "\n".join(
        [
            "You are a Claude Code powered Pokemon battle reviewer.",
            f"Model target: {model}.",
            "The deterministic Gen 1 custom bot has already scored this turn.",
            "Your job is to keep its decision unless there is a clear tactical reason to override it.",
            "Only choose one of the legal Showdown choices in `legal_choices`.",
            "Do not use sleep/paralysis/poison status moves into an opponent that already has a major status.",
            "Do not invent protocol actions. Do not explain outside JSON.",
            "Return exactly this JSON shape:",
            '{"schema_version":"pokerena.decision.v1","decision":"move 1","notes":"short rationale"}',
            "",
            "CUSTOM BOT BASELINE AND TURN DATA:",
            json.dumps(payload, indent=2, sort_keys=True),
        ]
    )


def _battle_summary(
    context_payload: Dict[str, Any],
    capture_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    public_lines = _public_lines(context_payload, capture_payload)
    own_active = _active_side_pokemon(_context_side(context_payload))
    opponent = _active_opponent_from_public_history(context_payload, public_lines)
    opponent_status = opponent.status if opponent is not None else None
    status_warning = None
    if opponent_status in MAJOR_STATUSES:
        status_warning = (
            f"Opponent already has major status {opponent_status}; "
            "do not override to sleep/paralysis/poison status moves."
        )
    return {
        "own_active": _pokemon_summary(own_active),
        "opponent_active": _pokemon_summary(opponent),
        "opponent_major_status": opponent_status,
        "status_warning": status_warning,
    }


def _pokemon_summary(pokemon: Any) -> Optional[Dict[str, Any]]:
    if pokemon is None:
        return None
    return {
        "species": pokemon.species,
        "ident": pokemon.ident,
        "hp_fraction": pokemon.hp_fraction,
        "status": pokemon.status,
        "volatile_statuses": list(pokemon.volatile_statuses),
    }


def _override_rejection_reason(
    context_payload: Dict[str, Any],
    capture_payload: Optional[Dict[str, Any]],
    decision: str,
) -> Optional[str]:
    public_lines = _public_lines(context_payload, capture_payload)
    opponent = _active_opponent_from_public_history(context_payload, public_lines)
    if opponent is None or opponent.status not in MAJOR_STATUSES:
        return None
    move_id = _decision_move_id(context_payload, decision)
    if move_id is None:
        return None
    if move_id in REDUNDANT_MAJOR_STATUS_MOVE_IDS:
        return f"opponent already has major status {opponent.status}"
    return None


def _decision_move_id(context_payload: Dict[str, Any], decision: str) -> Optional[str]:
    parts = decision.strip().split()
    if len(parts) != 2 or parts[0] != "move":
        return None
    try:
        index = int(parts[1])
    except ValueError:
        return None
    if index < 1:
        return None
    request = context_payload.get("request")
    active = request.get("active") if isinstance(request, dict) else None
    if not isinstance(active, list) or len(active) != 1 or not isinstance(active[0], dict):
        return None
    moves = active[0].get("moves")
    if not isinstance(moves, list) or index > len(moves):
        return None
    move = moves[index - 1]
    if not isinstance(move, dict):
        return None
    move_name = str(move.get("id") or move.get("move") or "")
    return _normalize_move_id(move_name) if move_name else None


def _baseline_with_note(baseline: AgentDecision, note: str) -> AgentDecision:
    return AgentDecision(
        schema_version=baseline.schema_version,
        decision=baseline.decision,
        notes=_join_notes(baseline.notes, note),
        raw_output=baseline.raw_output,
    )


def _join_notes(left: str, right: str) -> str:
    parts = [part.strip() for part in (left, right) if part and part.strip()]
    return " | ".join(parts)


def _legal_choices(context_payload: Dict[str, Any]) -> set[str]:
    choices: set[str] = set()
    hints = context_payload.get("legal_action_hints")
    if isinstance(hints, list):
        choices.update(str(item).strip() for item in hints if isinstance(item, str) and item.strip())
    request = context_payload.get("request")
    if not isinstance(request, dict):
        return choices
    active = request.get("active")
    if isinstance(active, list) and len(active) == 1 and isinstance(active[0], dict):
        moves = active[0].get("moves")
        if isinstance(moves, list):
            for index, move in enumerate(moves, start=1):
                if isinstance(move, dict) and not move.get("disabled", False):
                    choices.add(f"move {index}")
    if _request_allows_switches(request):
        side = request.get("side")
        pokemon = side.get("pokemon") if isinstance(side, dict) else None
        if isinstance(pokemon, list):
            for index, item in enumerate(pokemon, start=1):
                if not isinstance(item, dict) or item.get("active"):
                    continue
                if "fnt" not in str(item.get("condition") or ""):
                    choices.add(f"switch {index}")
    return choices


def _request_allows_switches(request: Dict[str, Any]) -> bool:
    force_switch = request.get("forceSwitch")
    if isinstance(force_switch, list) and any(item is True for item in force_switch):
        return True
    active = request.get("active")
    return not (
        isinstance(active, list)
        and len(active) == 1
        and isinstance(active[0], dict)
        and active[0].get("trapped") is True
    )


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"Unable to read {label} at {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"Unable to parse {label} JSON at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigError(f"{label} must be a JSON object.")
    return payload
