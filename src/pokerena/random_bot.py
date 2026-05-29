"""Simple stochastic Gen 1 randbat baseline bot."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, Optional

from .agent import AgentDecision, choose_random_legal
from .config import ConfigError
from .custom_bot import _available_switches, _single_active_request


DEFAULT_VOLUNTARY_SWITCH_CHANCE = 0.10


def decide_random_bot_from_files(
    *,
    context_path: Optional[str],
    capture_path: Optional[str],
    seed: Optional[str],
    switch_chance: float = DEFAULT_VOLUNTARY_SWITCH_CHANCE,
) -> AgentDecision:
    _ = capture_path
    resolved_context_path = Path(
        context_path or os.environ.get("POKERENA_TURN_CONTEXT_PATH") or ""
    )
    if not resolved_context_path.exists():
        raise ConfigError("Random bot requires --context or POKERENA_TURN_CONTEXT_PATH.")
    context_payload = _read_json_object(resolved_context_path, "turn context")
    rng = random.Random(seed) if seed is not None else random.Random()
    return build_random_bot_decision(context_payload, rng=rng, switch_chance=switch_chance)


def emit_random_bot_decision(decision: AgentDecision) -> None:
    print(json.dumps(asdict(decision), separators=(",", ":")))


def build_random_bot_decision(
    context: Dict[str, Any],
    *,
    rng: random.Random,
    switch_chance: float = DEFAULT_VOLUNTARY_SWITCH_CHANCE,
) -> AgentDecision:
    request = context.get("request") if isinstance(context.get("request"), dict) else None
    if not request:
        return _decision("move 1", "random-bot fallback: no active request.")

    if str(context.get("request_kind") or "") == "move":
        switches = _available_switches(request)
        if switches and rng.random() < max(0.0, min(1.0, switch_chance)):
            choice, label, _ = rng.choice(switches)
            return _decision(choice, f"random-bot selected voluntary switch at {switch_chance:.0%}: {label}.")
        moves = _enabled_move_choices(context)
        if moves:
            choice = rng.choice(moves)
            return _decision(choice, "random-bot selected a random enabled move.")

    choice = choose_random_legal(request, rng=rng)
    return _decision(choice, "random-bot fallback: selected random legal action.")


def _enabled_move_choices(context: Dict[str, Any]) -> list[str]:
    active_request = _single_active_request(context)
    moves = active_request.get("moves") if isinstance(active_request, dict) else None
    if not isinstance(moves, list):
        return []
    choices: list[str] = []
    for index, move in enumerate(moves, start=1):
        if isinstance(move, dict) and not move.get("disabled", False):
            choices.append(f"move {index}")
    return choices


def _decision(choice: str, notes: str) -> AgentDecision:
    return AgentDecision(
        schema_version="pokerena.decision.v1",
        decision=choice,
        notes=notes,
        raw_output="",
    )


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"Failed to read {label} JSON from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigError(f"{label.capitalize()} at {path} must contain a JSON object.")
    return payload
