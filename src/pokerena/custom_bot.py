"""Baseline heuristic Gen 1 random battle bot.

This is intentionally a stateless, beginner-to-intermediate strength bot rather
than a solved Gen 1 engine. It values damage, status, setup, and obvious pivots,
but it is still weak to meta-heavy lines such as Wrap locks, Evasion abuse, and
long-term sacrifice planning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .agent import AgentDecision, choose_first_legal, choose_random_legal
from .calc import (
    CALC_BATCH_REQUEST_SCHEMA_VERSION,
    CALC_REQUEST_SCHEMA_VERSION,
    DEFAULT_CALC_TIMEOUT_SECONDS,
    describe_move_metadata,
    detect_project_root,
    run_damage_calc_batch,
)
from .config import ConfigError


CUSTOM_BOT_SUPPORTED_FORMAT = "gen1randombattle"
SELECTION_WEIGHTED_SQUARE = "weighted-square"
SELECTION_ARGMAX = "argmax"
SELECTION_WEIGHTED_CUBE = "weighted-cube"
SELECTION_WEIGHTED_LINEAR = "weighted-linear"
DEFAULT_CUSTOM_BOT_SELECTION_STRATEGY = SELECTION_WEIGHTED_SQUARE
CUSTOM_BOT_SELECTION_STRATEGIES = (
    SELECTION_WEIGHTED_SQUARE,
    SELECTION_ARGMAX,
    SELECTION_WEIGHTED_CUBE,
    SELECTION_WEIGHTED_LINEAR,
)
MAJOR_STATUSES = {"slp", "par", "psn", "tox", "brn", "frz"}
SLEEP_MOVES = {"hypnosis", "lovelykiss", "sing", "sleeppowder", "spore"}
CHARGE_MOVE_IDS = {"solarbeam", "skyattack", "skullbash"}
DEFAULT_CHARGE_MOVE_MULTIPLIER = 0.45
CHARGE_MOVE_MULTIPLIERS = {"skyattack": 0.12}
SELF_KO_MOVE_IDS = {"explosion", "selfdestruct"}
RECOVERY_MOVE_IDS = {"recover", "softboiled"}
BOOST_CAP = 6
SLEEP_SOURCE_REST = "rest_sleep"
SLEEP_SOURCE_OPPONENT = "opponent_sleep"
SLEEP_SOURCE_UNKNOWN = "unknown_sleep"
GEN1_TYPE_IMMUNITIES = {
    "Electric": {"Ground"},
    "Fighting": {"Ghost"},
    "Ghost": {"Normal", "Psychic"},
    "Ground": {"Flying"},
    "Normal": {"Ghost"},
}
_POKEDEX_BASE_SPEED_CACHE: Dict[Path, Dict[str, int]] = {}
_POKEDEX_TYPES_CACHE: Dict[Path, Dict[str, List[str]]] = {}


@dataclass(frozen=True)
class ScoredAction:
    choice: str
    label: str
    score: float
    weight: float
    reason: str


@dataclass(frozen=True)
class CustomBotPlan:
    decision: str
    notes: str
    actions: List[ScoredAction]
    fallback_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DamageBatchResult:
    results: Dict[str, Dict[str, Any]]
    warning: Optional[str] = None


@dataclass(frozen=True)
class PokemonState:
    species: str
    ident: Optional[str]
    calc_ref: Dict[str, Any]
    hp_fraction: Optional[float]
    status: Optional[str]
    volatile_statuses: tuple[str, ...] = ()


@dataclass(frozen=True)
class MoveCandidate:
    choice: str
    index: int
    name: str
    request_move: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class SwitchCandidateEvaluation:
    score: float
    reason: str
    faster: Optional[bool] = None
    reliable_ko: bool = False


def decide_custom_bot_from_files(
    *,
    context_path: Optional[str],
    capture_path: Optional[str],
    seed: Optional[str],
    project_root: Optional[Path] = None,
    calc_timeout_seconds: int = DEFAULT_CALC_TIMEOUT_SECONDS,
    selection_strategy: str = DEFAULT_CUSTOM_BOT_SELECTION_STRATEGY,
) -> AgentDecision:
    resolved_context_path = Path(
        context_path or os.environ.get("POKERENA_TURN_CONTEXT_PATH") or ""
    )
    if not resolved_context_path.exists():
        raise ConfigError("Custom bot requires --context or POKERENA_TURN_CONTEXT_PATH.")
    context_payload = _read_json_object(resolved_context_path, "turn context")

    capture_value = capture_path or os.environ.get("POKERENA_BATTLE_CAPTURE_PATH")
    resolved_capture_path = Path(capture_value) if capture_value else None
    capture_payload = (
        _read_json_object(resolved_capture_path, "battle capture")
        if resolved_capture_path is not None and resolved_capture_path.exists()
        else None
    )
    rng = random.Random(seed) if seed is not None else random.Random()
    plan = build_custom_bot_plan(
        context_payload,
        capture_payload=capture_payload,
        project_root=project_root,
        rng=rng,
        calc_timeout_seconds=calc_timeout_seconds,
        selection_strategy=selection_strategy,
    )
    for warning in plan.warnings or []:
        print(warning, file=sys.stderr)
    return AgentDecision(
        schema_version="pokerena.decision.v1",
        decision=plan.decision,
        notes=plan.notes,
        raw_output="",
    )


def emit_custom_bot_decision(decision: AgentDecision) -> None:
    print(json.dumps(asdict(decision), separators=(",", ":")))


def build_custom_bot_plan(
    context: Dict[str, Any],
    *,
    capture_payload: Optional[Dict[str, Any]] = None,
    project_root: Optional[Path] = None,
    rng: Optional[random.Random] = None,
    calc_timeout_seconds: int = DEFAULT_CALC_TIMEOUT_SECONDS,
    selection_strategy: str = DEFAULT_CUSTOM_BOT_SELECTION_STRATEGY,
) -> CustomBotPlan:
    chooser = rng or random.Random()
    strategy = _normalize_selection_strategy(selection_strategy)
    request = context.get("request") if isinstance(context.get("request"), dict) else None
    if not request:
        return _fallback_plan(request, "no active request", chooser)

    request_kind = str(context.get("request_kind") or "")
    if request_kind == "switch":
        root = project_root or detect_project_root()
        public_lines = _public_lines(context, capture_payload)
        opponent = _active_opponent_from_public_history(context, public_lines)
        return _forced_switch_plan(
            request,
            chooser,
            context=context,
            opponent=opponent,
            public_lines=public_lines,
            project_root=root,
            calc_timeout_seconds=calc_timeout_seconds,
            selection_strategy=strategy,
        )

    if request_kind != "move" or not _is_supported_gen1_randbat(context):
        return _fallback_plan(request, "unsupported format or request kind", chooser)

    active_request = _single_active_request(context)
    own_active = _active_side_pokemon(_context_side(context))
    public_lines = _public_lines(context, capture_payload)
    opponent = _active_opponent_from_public_history(context, public_lines)
    if active_request is None or own_active is None or opponent is None:
        return _fallback_plan(request, "missing active Pokemon or opponent state", chooser)
    own_active = _with_volatile_statuses(
        own_active,
        _volatile_statuses_for_active(public_lines, own_active.ident, own_active.volatile_statuses),
    )

    root = project_root or detect_project_root()
    move_candidates = _move_candidates(
        active_request=active_request,
        project_root=root,
        calc_timeout_seconds=calc_timeout_seconds,
    )
    if not move_candidates:
        return _fallback_plan(request, "no enabled moves", chooser)

    damage_results = _damage_results_by_choice(
        candidates=move_candidates,
        attacker=own_active,
        defender=opponent,
        project_root=root,
        calc_timeout_seconds=calc_timeout_seconds,
    )
    damage_warnings = [
        damage_results.warning,
    ] if damage_results.warning else []
    sleep_clause_active = _sleep_clause_active(context, public_lines)
    own_boosts = _boosts_for_active(context, public_lines, own_active.ident)
    active_sleep_source = _active_sleep_source(context, public_lines, own_active)
    reliable_ko_choices = _reliable_ko_choices(damage_results.results, opponent)
    accurate_reliable_ko_choices = _accurate_reliable_ko_choices(
        candidates=move_candidates,
        damage_results=damage_results.results,
        opponent=opponent,
    )
    has_ko_line = _has_ko_line(damage_results.results.values(), opponent)
    mon_count_advantage = _remaining_pokemon_advantage(context, public_lines)
    opponent_moves = _revealed_opponent_moves(context, public_lines)[:4]
    counter_revealed = _counter_revealed(opponent_moves)
    incoming_threat_pct = _worst_incoming_damage_pct(
        opponent=opponent,
        defender=own_active,
        move_names=opponent_moves,
        project_root=root,
        calc_timeout_seconds=calc_timeout_seconds,
    )
    action_scores: List[ScoredAction] = []
    best_damage_score = 0.0
    for candidate in move_candidates:
        damage_result = damage_results.results.get(candidate.choice)
        if damage_result is not None:
            score, reason = _score_damaging_move(
                candidate,
                damage_result,
                own_active=own_active,
                opponent=opponent,
                mon_count_advantage=mon_count_advantage,
                counter_revealed=counter_revealed,
                project_root=root,
            )
            score = _add_secondary_status_value(score, candidate, opponent)
        elif _is_damaging_metadata(candidate.metadata):
            score, reason = _score_uncalculated_damaging_move(
                candidate,
                own_active=own_active,
                opponent=opponent,
                project_root=root,
                counter_revealed=counter_revealed,
            )
        else:
            score, reason = _score_status_or_utility_move(
                candidate,
                own_active=own_active,
                opponent=opponent,
                own_boosts=own_boosts,
                sleep_clause_active=sleep_clause_active,
                has_ko_line=has_ko_line,
                has_reliable_ko_line=bool(reliable_ko_choices),
                incoming_threat_pct=incoming_threat_pct,
                project_root=root,
            )
        if (
            score > 0
            and accurate_reliable_ko_choices
            and candidate.choice not in accurate_reliable_ko_choices
            and _is_damaging_metadata(candidate.metadata)
            and _accuracy_factor(candidate.metadata) < 1.0
        ):
            score *= 0.10
            reason = f"{reason}; inaccurate move heavily deprioritized: 100% accurate KO available"
        if score > 0 and own_active.status == "slp":
            score *= 0.20
            reason = f"{reason}; active asleep"
        if _is_damaging_metadata(candidate.metadata):
            best_damage_score = max(best_damage_score, score)
        if score > 0:
            action_scores.append(_scored_action(candidate.choice, candidate.name, score, reason))

    active_has_reliable_ko = False
    if reliable_ko_choices:
        reliable_ko_scores = [
            action for action in action_scores if action.choice in reliable_ko_choices and action.score >= 20.0
        ]
        if reliable_ko_scores:
            action_scores = reliable_ko_scores
            active_has_reliable_ko = own_active.status not in {"slp", "frz"}

    force_sleep_switch = own_active.status == "slp" and active_sleep_source != SLEEP_SOURCE_REST
    force_switch_only = own_active.status == "frz" or force_sleep_switch or not action_scores
    major_setup_locked = _major_setup_lock_in_active(
        own_active=own_active,
        own_boosts=own_boosts,
        action_scores=action_scores,
        incoming_threat_pct=incoming_threat_pct,
    )

    switch_scores = _voluntary_switch_scores(
        context=context,
        request=request,
        own_active=own_active,
        opponent=opponent,
        public_lines=public_lines,
        existing_scores=action_scores,
        best_damage_score=best_damage_score,
        own_boosts=own_boosts,
        active_has_reliable_ko=active_has_reliable_ko,
        force_switch_only=force_switch_only,
        active_sleep_source=active_sleep_source,
        major_setup_locked=major_setup_locked,
        project_root=root,
        calc_timeout_seconds=calc_timeout_seconds,
    )
    if force_switch_only and switch_scores:
        decision, selection_pool = _select_action(_selection_pool(switch_scores), chooser, strategy)
        notes = _notes(
            decision,
            selection_pool,
            fallback_reason=None,
            warnings=damage_warnings,
            selection_strategy=strategy,
        )
        return CustomBotPlan(decision=decision, notes=notes, actions=selection_pool, warnings=damage_warnings)
    if force_switch_only and not switch_scores:
        emergency_switches = _least_bad_switch_scores(
            request=request,
            opponent=opponent,
            opponent_moves=opponent_moves,
            project_root=root,
            calc_timeout_seconds=calc_timeout_seconds,
            reason_prefix="emergency switch fallback",
        )
        if emergency_switches:
            decision, selection_pool = _select_action(_selection_pool(emergency_switches), chooser, strategy)
            notes = _notes(
                decision,
                selection_pool,
                fallback_reason="all active moves scored zero",
                warnings=damage_warnings,
                selection_strategy=strategy,
            )
            return CustomBotPlan(
                decision=decision,
                notes=notes,
                actions=selection_pool,
                fallback_reason="all active moves scored zero",
                warnings=damage_warnings,
            )
    action_scores.extend(switch_scores)
    if not action_scores:
        safe_fallback = _safe_zero_score_fallback_choice(
            candidates=move_candidates,
            opponent=opponent,
            sleep_clause_active=sleep_clause_active,
            project_root=root,
        )
        return _fallback_plan(
            request,
            "all heuristic scores were zero",
            chooser,
            warnings=damage_warnings,
            preferred_decision=safe_fallback,
        )

    decision, selection_pool = _select_action(_selection_pool(action_scores), chooser, strategy)
    notes = _notes(
        decision,
        selection_pool,
        fallback_reason=None,
        warnings=damage_warnings,
        selection_strategy=strategy,
    )
    return CustomBotPlan(decision=decision, notes=notes, actions=selection_pool, warnings=damage_warnings)


def _forced_switch_plan(
    request: Dict[str, Any],
    rng: random.Random,
    *,
    context: Optional[Dict[str, Any]] = None,
    opponent: Optional[PokemonState] = None,
    public_lines: Sequence[str] = (),
    project_root: Optional[Path] = None,
    calc_timeout_seconds: int = DEFAULT_CALC_TIMEOUT_SECONDS,
    selection_strategy: str = DEFAULT_CUSTOM_BOT_SELECTION_STRATEGY,
) -> CustomBotPlan:
    strategy = _normalize_selection_strategy(selection_strategy)
    switches = _available_switches(request)
    if not switches:
        return _fallback_plan(request, "no legal switch targets", rng)
    opponent_moves = _revealed_opponent_moves(context, public_lines)[:4] if context else []
    actions: List[ScoredAction] = []
    for choice, label, pokemon in switches:
        if opponent is not None and project_root is not None:
            evaluation = _switch_candidate_evaluation(
                pokemon=pokemon,
                opponent=opponent,
                opponent_moves=opponent_moves,
                project_root=project_root,
                calc_timeout_seconds=calc_timeout_seconds,
            )
            if evaluation is None:
                continue
            score = evaluation.score
            reason = f"forced switch matchup; {evaluation.reason}"
        else:
            score = max(0.01, _condition_hp_fraction(pokemon.get("condition")) or 0.0) * 100.0
            reason = "forced switch"
        actions.append(_scored_action(choice, label, max(0.01, score), reason))
    if not actions:
        fallback_actions = (
            _least_bad_switch_scores(
                request=request,
                opponent=opponent,
                opponent_moves=opponent_moves,
                project_root=project_root,
                calc_timeout_seconds=calc_timeout_seconds,
                reason_prefix="forced switch fallback",
            )
            if opponent is not None and project_root is not None
            else []
        )
        if fallback_actions:
            decision, selection_pool = _select_action(_selection_pool(fallback_actions), rng, strategy)
            return CustomBotPlan(
                decision=decision,
                notes=_notes(
                    decision,
                    selection_pool,
                    fallback_reason="no viable switch targets",
                    selection_strategy=strategy,
                ),
                actions=selection_pool,
                fallback_reason="no viable switch targets",
            )
        return _fallback_plan(request, "no viable switch targets", rng)
    decision, selection_pool = _select_action(_selection_pool(actions), rng, strategy)
    return CustomBotPlan(
        decision=decision,
        notes=_notes(decision, selection_pool, fallback_reason=None, selection_strategy=strategy),
        actions=selection_pool,
    )


def _fallback_plan(
    request: Optional[Dict[str, Any]],
    reason: str,
    rng: random.Random,
    *,
    warnings: Optional[List[str]] = None,
    preferred_decision: Optional[str] = None,
) -> CustomBotPlan:
    decision = preferred_decision or (choose_random_legal(request, rng=rng) if request is not None else choose_first_legal(request))
    warning_prefix = " ".join(warnings or [])
    notes = f"{warning_prefix} custom-bot fallback: {reason}; selected {decision}.".strip()
    return CustomBotPlan(
        decision=decision,
        notes=notes,
        actions=[],
        fallback_reason=reason,
        warnings=list(warnings or []),
    )


def _is_supported_gen1_randbat(context: Dict[str, Any]) -> bool:
    format_name = str(context.get("format_name") or "").lower()
    normalized = re.sub(r"[^a-z0-9]+", "", format_name)
    return normalized == CUSTOM_BOT_SUPPORTED_FORMAT or (
        "gen1" in normalized and "random" in normalized and "battle" in normalized
    )


def _single_active_request(context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    active = context.get("active")
    if not isinstance(active, list) or len(active) != 1 or not isinstance(active[0], dict):
        return None
    return active[0]


def _context_side(context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    side = context.get("side")
    if isinstance(side, dict):
        return side
    request = context.get("request")
    if isinstance(request, dict) and isinstance(request.get("side"), dict):
        return request["side"]
    return None


def _move_candidates(
    *,
    active_request: Dict[str, Any],
    project_root: Path,
    calc_timeout_seconds: int,
) -> List[MoveCandidate]:
    moves = active_request.get("moves")
    if not isinstance(moves, list):
        return []
    enabled: List[tuple[int, str, Dict[str, Any]]] = []
    for index, move in enumerate(moves, start=1):
        if not isinstance(move, dict) or move.get("disabled", False):
            continue
        move_name = str(move.get("move") or move.get("id") or "").strip()
        if move_name:
            enabled.append((index, move_name, move))
    if not enabled:
        return []

    metadata_by_name: Dict[str, Dict[str, Any]] = {}
    try:
        payload = describe_move_metadata(
            project_root=project_root,
            generation=1,
            move_names=[name for _, name, _ in enabled],
            timeout_seconds=calc_timeout_seconds,
        )
        for item in payload.get("moves", []):
            if isinstance(item, dict):
                metadata_by_name[_normalize_move_id(str(item.get("requested_name") or item.get("name") or ""))] = item
    except ConfigError:
        metadata_by_name = {}

    candidates: List[MoveCandidate] = []
    for index, move_name, request_move in enabled:
        metadata = metadata_by_name.get(_normalize_move_id(move_name)) or _heuristic_move_metadata(
            move_name,
            request_move,
        )
        candidates.append(
            MoveCandidate(
                choice=f"move {index}",
                index=index,
                name=str(metadata.get("name") or move_name),
                request_move=request_move,
                metadata=metadata,
            )
        )
    return candidates


def _damage_results_by_choice(
    *,
    candidates: Sequence[MoveCandidate],
    attacker: PokemonState,
    defender: PokemonState,
    project_root: Path,
    calc_timeout_seconds: int,
) -> DamageBatchResult:
    requests: List[Dict[str, Any]] = []
    choices: List[str] = []
    for candidate in candidates:
        if not _is_damaging_metadata(candidate.metadata):
            continue
        requests.append(
            {
                "schema_version": CALC_REQUEST_SCHEMA_VERSION,
                "generation": 1,
                "attacker": attacker.calc_ref,
                "defender": defender.calc_ref,
                "move": {"name": candidate.name},
                "field": {},
            }
        )
        choices.append(candidate.choice)
    if not requests:
        return DamageBatchResult(results={})
    try:
        response = run_damage_calc_batch(
            {"schema_version": CALC_BATCH_REQUEST_SCHEMA_VERSION, "requests": requests},
            project_root=project_root,
            timeout_seconds=calc_timeout_seconds,
        )
    except ConfigError as error:
        return DamageBatchResult(
            results={},
            warning=(
                "custom-bot warning: damage calc batch failed; damaging moves are "
                f"using generic utility fallback ({error})."
            ),
        )
    results = response.get("results")
    if not isinstance(results, list):
        return DamageBatchResult(
            results={},
            warning="custom-bot warning: damage calc batch returned an invalid result shape.",
        )
    mapped: Dict[str, Dict[str, Any]] = {}
    for choice, result in zip(choices, results):
        if isinstance(result, dict) and result.get("status") == "ok" and isinstance(result.get("result"), dict):
            mapped[choice] = result["result"]
    return DamageBatchResult(results=mapped)


def _score_damaging_move(
    candidate: MoveCandidate,
    damage_result: Dict[str, Any],
    *,
    own_active: PokemonState,
    opponent: PokemonState,
    mon_count_advantage: int,
    counter_revealed: bool,
    project_root: Path,
) -> tuple[float, str]:
    if _move_has_no_effect(candidate, opponent, project_root):
        return 0.0, "damage move has no effect into target type"
    range_percent = damage_result.get("range_percent") if isinstance(damage_result, dict) else None
    if not isinstance(range_percent, dict):
        return 0.0, "missing damage range"
    min_pct = _float_value(range_percent.get("min")) or 0.0
    max_pct = _float_value(range_percent.get("max")) or 0.0
    target_hp_pct = 100.0 * (opponent.hp_fraction if opponent.hp_fraction is not None else 1.0)
    # This intentionally uses the calc range midpoint as a lightweight approximation
    # rather than modeling Gen 1's exact 217-255 damage roll distribution.
    expected_pct = (min_pct + max_pct) / 2.0
    score = min(expected_pct, target_hp_pct) * _accuracy_factor(candidate.metadata)
    if min_pct >= target_hp_pct:
        score += 30.0
    elif max_pct >= target_hp_pct:
        score += 15.0
    if bool(candidate.metadata.get("high_crit")):
        score *= 1.05
    move_id = _metadata_id(candidate.metadata)
    if move_id == "hyperbeam":
        hyper_beam_multiplier = _hyper_beam_recharge_risk_multiplier(
            min_pct=min_pct,
            max_pct=max_pct,
            target_hp_pct=target_hp_pct,
        )
        score *= hyper_beam_multiplier
    charge_multiplier = _charge_move_multiplier(move_id, candidate.metadata)
    if charge_multiplier < 1.0:
        score *= charge_multiplier
    reason = f"damage {min_pct:.1f}-{max_pct:.1f}%"
    if charge_multiplier < 1.0:
        reason = (
            f"{reason}; Sky Attack heavily deprioritized"
            if move_id == "skyattack"
            else f"{reason}; charge turn penalty"
        )
    if counter_revealed and _move_type(candidate.metadata) in {"Normal", "Fighting"}:
        score *= 0.25
        reason = f"{reason}; Counter risk"
    if move_id in SELF_KO_MOVE_IDS:
        multiplier, self_ko_reason = _self_ko_move_multiplier(
            own_active=own_active,
            min_pct=min_pct,
            max_pct=max_pct,
            target_hp_pct=target_hp_pct,
            mon_count_advantage=mon_count_advantage,
        )
        score *= multiplier
        reason = f"{reason}; {self_ko_reason}"
    return max(0.0, score), reason


def _self_ko_move_multiplier(
    *,
    own_active: PokemonState,
    min_pct: float,
    max_pct: float,
    target_hp_pct: float,
    mon_count_advantage: int,
) -> tuple[float, str]:
    own_hp_fraction = own_active.hp_fraction if own_active.hp_fraction is not None else 1.0
    if own_hp_fraction <= 0.30:
        return 1.0, "self-KO allowed: user low HP"
    reliable_ko = min_pct >= target_hp_pct
    possible_ko = max_pct >= target_hp_pct
    if reliable_ko and mon_count_advantage > 0:
        return 1.0, "self-KO allowed: KO while ahead in mons"
    if reliable_ko:
        return 0.05, "self-KO heavily deprioritized: KO trade not ahead in mons"
    if possible_ko:
        return 0.03, "self-KO heavily deprioritized: uncertain KO from healthy user"
    return 0.0, "self-KO blocked: no KO from healthy user"


def _score_uncalculated_damaging_move(
    candidate: MoveCandidate,
    *,
    own_active: PokemonState,
    opponent: PokemonState,
    project_root: Path,
    counter_revealed: bool,
) -> tuple[float, str]:
    if _move_has_no_effect(candidate, opponent, project_root):
        return 0.0, "damage move has no effect into target type"
    base_power = _int_value(candidate.metadata.get("base_power"), 0)
    if base_power <= 0:
        return 0.0, "damage calc skipped"
    score = base_power * 0.25 * _accuracy_factor(candidate.metadata)
    move_type = _move_type(candidate.metadata)
    if move_type and move_type in _pokemon_types(own_active, project_root):
        score *= 1.5
    reason = "estimated damage: calc unavailable"
    move_id = _metadata_id(candidate.metadata)
    charge_multiplier = _charge_move_multiplier(move_id, candidate.metadata)
    if charge_multiplier < 1.0:
        score *= charge_multiplier
        reason = (
            f"{reason}; Sky Attack heavily deprioritized"
            if move_id == "skyattack"
            else f"{reason}; charge turn penalty"
        )
    if counter_revealed and move_type in {"Normal", "Fighting"}:
        score *= 0.25
        reason = f"{reason}; Counter risk"
    if move_id in SELF_KO_MOVE_IDS:
        own_hp_fraction = own_active.hp_fraction if own_active.hp_fraction is not None else 1.0
        if own_hp_fraction > 0.30:
            return 0.0, "estimated damage: self-KO blocked while calc unavailable"
        else:
            reason = "estimated damage: self-KO allowed at low HP"
    return max(1.0, min(score, 35.0)), reason


def _has_ko_line(
    damage_results: Iterable[Dict[str, Any]],
    opponent: PokemonState,
) -> bool:
    target_hp_pct = 100.0 * (opponent.hp_fraction if opponent.hp_fraction is not None else 1.0)
    for damage_result in damage_results:
        range_percent = damage_result.get("range_percent") if isinstance(damage_result, dict) else None
        if not isinstance(range_percent, dict):
            continue
        max_pct = _float_value(range_percent.get("max")) or 0.0
        if max_pct >= target_hp_pct:
            return True
    return False


def _safe_zero_score_fallback_choice(
    *,
    candidates: Sequence[MoveCandidate],
    opponent: PokemonState,
    sleep_clause_active: bool,
    project_root: Path,
) -> Optional[str]:
    for candidate in candidates:
        move_id = _metadata_id(candidate.metadata)
        if move_id in SELF_KO_MOVE_IDS:
            continue
        if _is_damaging_metadata(candidate.metadata):
            if _move_has_no_effect(candidate, opponent, project_root):
                continue
            return candidate.choice
        status = str(candidate.metadata.get("status") or "")
        if status in MAJOR_STATUSES:
            if opponent.status in MAJOR_STATUSES:
                continue
            if status == "slp" and sleep_clause_active:
                continue
            if _status_move_is_ineffective(candidate, opponent, project_root):
                continue
            return candidate.choice
        volatile_status = str(candidate.metadata.get("volatile_status") or "")
        if volatile_status == "confusion" and _has_volatile_status(opponent, "confusion"):
            continue
        return candidate.choice
    return None


def _reliable_ko_choices(
    damage_results: Dict[str, Dict[str, Any]],
    opponent: PokemonState,
) -> set[str]:
    target_hp_pct = 100.0 * (opponent.hp_fraction if opponent.hp_fraction is not None else 1.0)
    choices: set[str] = set()
    for choice, damage_result in damage_results.items():
        range_percent = damage_result.get("range_percent") if isinstance(damage_result, dict) else None
        if not isinstance(range_percent, dict):
            continue
        min_pct = _float_value(range_percent.get("min")) or 0.0
        if min_pct >= target_hp_pct:
            choices.add(choice)
    return choices


def _accurate_reliable_ko_choices(
    *,
    candidates: Sequence[MoveCandidate],
    damage_results: Dict[str, Dict[str, Any]],
    opponent: PokemonState,
) -> set[str]:
    reliable_choices = _reliable_ko_choices(damage_results, opponent)
    return {
        candidate.choice
        for candidate in candidates
        if candidate.choice in reliable_choices and _accuracy_factor(candidate.metadata) >= 1.0
    }


def _hyper_beam_recharge_risk_multiplier(
    *,
    min_pct: float,
    max_pct: float,
    target_hp_pct: float,
) -> float:
    if min_pct >= target_hp_pct:
        return 1.0
    if max_pct <= target_hp_pct:
        return 0.10
    roll_span = max_pct - min_pct
    if roll_span <= 0:
        return 0.10
    ko_fraction = max(0.0, min(1.0, (max_pct - target_hp_pct) / roll_span))
    return 0.20 + (ko_fraction * 0.80)


def _charge_move_multiplier(move_id: str, metadata: Dict[str, Any]) -> float:
    if not (bool(metadata.get("charge")) or move_id in CHARGE_MOVE_IDS):
        return 1.0
    return CHARGE_MOVE_MULTIPLIERS.get(move_id, DEFAULT_CHARGE_MOVE_MULTIPLIER)


def _add_secondary_status_value(
    score: float,
    candidate: MoveCandidate,
    opponent: PokemonState,
) -> float:
    if opponent.status in MAJOR_STATUSES:
        return score
    secondary = candidate.metadata.get("secondary")
    if not isinstance(secondary, dict):
        return score
    chance = _float_value(secondary.get("chance"))
    status = str(secondary.get("status") or "")
    if chance is None or status not in MAJOR_STATUSES:
        return score
    return score + (
        _status_base_value(status, sleep_clause_active=False) * (chance / 100.0) * 0.35
    )


def _score_status_or_utility_move(
    candidate: MoveCandidate,
    *,
    own_active: PokemonState,
    opponent: PokemonState,
    own_boosts: Dict[str, int],
    sleep_clause_active: bool,
    has_ko_line: bool,
    has_reliable_ko_line: bool,
    incoming_threat_pct: Optional[float],
    project_root: Path,
) -> tuple[float, str]:
    metadata = candidate.metadata
    status = str(metadata.get("status") or "")
    if status in MAJOR_STATUSES:
        if has_reliable_ko_line:
            return 0.0, f"inflict {status} blocked: reliable KO available"
        if opponent.status in MAJOR_STATUSES:
            return 0.0, f"target already statused with {opponent.status}"
        if status == "slp" and sleep_clause_active:
            return 0.0, "sleep clause active"
        if _status_move_is_ineffective(candidate, opponent, project_root):
            return 0.0, "status move is ineffective into target type"
        score = _status_base_value(status, sleep_clause_active) * _accuracy_factor(metadata)
        if has_ko_line:
            return score * 0.05, f"inflict {status} heavily deprioritized: possible KO available"
        return score, f"inflict {status}"

    move_id = _metadata_id(metadata)
    hp_fraction = own_active.hp_fraction if own_active.hp_fraction is not None else 1.0
    if move_id == "reflect" and _has_volatile_status(own_active, "reflect"):
        return 0.0, "Reflect already active"
    boosts = metadata.get("boosts")
    if isinstance(boosts, dict) and boosts:
        if move_id == "barrier" and own_boosts.get("def", 0) > 0:
            return 0.0, "Barrier already active"
        if move_id == "agility" and "spe" in boosts:
            score, reason = _agility_score(
                own_active=own_active,
                opponent=opponent,
                own_boosts=own_boosts,
                hp_fraction=hp_fraction,
                project_root=project_root,
            )
            score, reason = _adjust_setup_score(
                score,
                reason,
                own_active=own_active,
                has_reliable_ko_line=has_reliable_ko_line,
                has_ko_line=has_ko_line,
                incoming_threat_pct=incoming_threat_pct,
            )
            return score, reason
        score = _boost_score(move_id, boosts, hp_fraction, own_boosts)
        score, reason = _adjust_setup_score(
            score,
            "setup boost",
            own_active=own_active,
            has_reliable_ko_line=has_reliable_ko_line,
            has_ko_line=has_ko_line,
            incoming_threat_pct=incoming_threat_pct,
        )
        return score, reason

    volatile_status = str(metadata.get("volatile_status") or "")
    if volatile_status == "leechseed" and opponent.status not in MAJOR_STATUSES:
        if has_reliable_ko_line:
            return 0.0, "Leech Seed pressure blocked: reliable KO available"
        score = 40.0 * _accuracy_factor(metadata)
        if has_ko_line:
            return score * 0.05, "Leech Seed pressure heavily deprioritized: possible KO available"
        return score, "Leech Seed pressure"
    if volatile_status == "confusion":
        if has_reliable_ko_line:
            return 0.0, "confusion pressure blocked: reliable KO available"
        if _has_volatile_status(opponent, "confusion"):
            return 0.0, "target already confused"
        score = 35.0 * _accuracy_factor(metadata)
        if has_ko_line:
            return score * 0.05, "confusion pressure heavily deprioritized: possible KO available"
        return score, "confusion pressure"
    if volatile_status in {"reflect", "lightscreen"}:
        if _has_volatile_status(own_active, volatile_status):
            return 0.0, f"{volatile_status} already active"
        if has_reliable_ko_line:
            return 0.0, f"{volatile_status} blocked: reliable KO available"
        return (35.0 * hp_fraction) if hp_fraction >= 0.35 else 5.0, volatile_status
    if move_id in RECOVERY_MOVE_IDS:
        if hp_fraction <= 0.65:
            return max(8.0, 85.0 * (1.0 - hp_fraction)), "recovery"
        if hp_fraction <= 0.80:
            return 12.0 * (1.0 - hp_fraction), "recovery"
        return 2.0, "recovery"
    if move_id == "rest":
        if own_active.status in {"par", "psn", "tox", "brn"} and hp_fraction < 0.50:
            return max(18.0, 42.0 * (1.0 - hp_fraction)), "Rest status-curing recovery with sleep cost"
        if own_active.status in {"slp", "frz"}:
            return 0.0, f"Rest blocked: already {own_active.status}"
        if hp_fraction < 0.25:
            return max(10.0, 30.0 * (1.0 - hp_fraction)), "Rest emergency recovery with sleep cost"
        return 0.0, "Rest too costly at current HP"
    return 5.0 * _accuracy_factor(metadata), "generic utility"


def _status_base_value(status: str, sleep_clause_active: bool) -> float:
    if status == "slp":
        return 0.0 if sleep_clause_active else 90.0
    if status == "par":
        return 70.0
    if status == "frz":
        return 88.0
    if status in {"psn", "tox", "brn"}:
        return 45.0
    return 20.0


def _status_move_is_ineffective(
    candidate: MoveCandidate,
    opponent: PokemonState,
    project_root: Path,
) -> bool:
    move_id = _metadata_id(candidate.metadata)
    if move_id != "thunderwave":
        return False
    return "Ground" in _pokemon_types(opponent, project_root)


def _move_has_no_effect(
    candidate: MoveCandidate,
    opponent: PokemonState,
    project_root: Path,
) -> bool:
    move_type = _move_type(candidate.metadata)
    if not move_type:
        return False
    immune_types = GEN1_TYPE_IMMUNITIES.get(move_type)
    if not immune_types:
        return False
    return any(pokemon_type in immune_types for pokemon_type in _pokemon_types(opponent, project_root))


def _move_type(metadata: Dict[str, Any]) -> Optional[str]:
    move_type = metadata.get("type")
    if not isinstance(move_type, str) or not move_type.strip():
        return None
    return move_type.strip().title()


def _boost_score(
    move_id: str,
    boosts: Dict[str, Any],
    hp_fraction: float,
    own_boosts: Dict[str, int],
) -> float:
    if _boosts_are_capped(boosts, own_boosts):
        return 0.0
    if move_id == "amnesia":
        if hp_fraction >= 0.70:
            return 90.0 * hp_fraction
        if hp_fraction >= 0.40:
            return 55.0 * hp_fraction
        return 15.0 * hp_fraction
    if "atk" in boosts:
        return 60.0 * hp_fraction if hp_fraction >= 0.45 else 10.0 * hp_fraction
    if "spe" in boosts:
        return 45.0 * hp_fraction if hp_fraction >= 0.45 else 8.0 * hp_fraction
    if "evasion" in boosts:
        return 75.0 * hp_fraction if hp_fraction >= 0.40 else 18.0 * hp_fraction
    return 35.0 * hp_fraction


def _agility_score(
    *,
    own_active: PokemonState,
    opponent: PokemonState,
    own_boosts: Dict[str, int],
    hp_fraction: float,
    project_root: Path,
) -> tuple[float, str]:
    if own_boosts.get("spe", 0) > 0:
        return 0.0, "Agility already boosted Speed"
    own_speed = _known_speed_stat(own_active) or _estimated_gen1_speed(own_active, project_root)
    opponent_speed = _effective_gen1_speed(opponent, project_root)
    if own_speed is not None and opponent_speed is not None:
        current_speed = _effective_speed_value(
            own_speed,
            status=own_active.status,
            speed_boost=own_boosts.get("spe", 0),
        )
        post_agility_speed = _effective_speed_value(own_speed, status=None, speed_boost=2)
        if current_speed is not None and current_speed >= opponent_speed:
            return 0.0, "already likely faster"
        if post_agility_speed <= opponent_speed:
            return 0.0, "Agility would not outspeed opponent"
    if own_active.status == "par":
        if hp_fraction >= 0.35:
            return 92.0 * max(hp_fraction, 0.55), "Agility offsets Gen 1 paralysis Speed loss"
        return 30.0 * hp_fraction, "Agility offsets paralysis but user is low HP"

    if own_speed is None or opponent_speed is None:
        if hp_fraction >= 0.70:
            return 22.0 * hp_fraction, "Agility with unknown Speed matchup"
        if hp_fraction >= 0.45:
            return 12.0 * hp_fraction, "Agility with unknown Speed matchup"
        return 3.0 * hp_fraction, "Agility with unknown Speed matchup"

    if own_speed >= opponent_speed:
        return 0.0, "already likely faster"
    if hp_fraction >= 0.75:
        return 72.0 * hp_fraction, "Agility to outspeed likely faster opponent"
    if hp_fraction >= 0.45:
        return 38.0 * hp_fraction, "Agility to outspeed likely faster opponent"
    return 8.0 * hp_fraction, "Agility too risky at low HP"


def _adjust_setup_score(
    score: float,
    reason: str,
    *,
    own_active: PokemonState,
    has_reliable_ko_line: bool,
    has_ko_line: bool,
    incoming_threat_pct: Optional[float],
) -> tuple[float, str]:
    if score <= 0:
        return score, reason
    if has_reliable_ko_line:
        return 0.0, f"{reason} blocked: reliable KO available"
    if has_ko_line:
        score *= 0.05
        reason = f"{reason}; setup heavily deprioritized: possible KO available"
    own_hp_pct = 100.0 * (own_active.hp_fraction if own_active.hp_fraction is not None else 1.0)
    if incoming_threat_pct is None or own_hp_pct <= 0:
        return score, reason
    if incoming_threat_pct >= own_hp_pct:
        return 0.0, f"{reason}; setup blocked: revealed attack can KO"
    if incoming_threat_pct >= own_hp_pct * 0.75:
        return score * 0.15, f"{reason}; setup heavily deprioritized: revealed attack is dangerous"
    if incoming_threat_pct >= own_hp_pct * 0.50:
        return score * 0.45, f"{reason}; setup deprioritized: revealed attack is threatening"
    return score, reason


def _boosts_are_capped(boosts: Dict[str, Any], own_boosts: Dict[str, int]) -> bool:
    checked = False
    for stat, delta in boosts.items():
        if not isinstance(stat, str):
            continue
        if isinstance(delta, bool) or not isinstance(delta, int) or delta <= 0:
            continue
        checked = True
        if own_boosts.get(stat, 0) + delta <= BOOST_CAP:
            return False
    return checked


def _has_positive_boost(boosts: Dict[str, int]) -> bool:
    return any(value > 0 for value in boosts.values())


def _positive_boost_total(boosts: Dict[str, int]) -> int:
    return sum(max(0, value) for value in boosts.values())


def _has_major_setup(boosts: Dict[str, int]) -> bool:
    return boosts.get("spa", 0) >= 2 or _positive_boost_total(boosts) >= 2


def _has_meaningful_active_path(action_scores: Sequence[ScoredAction]) -> bool:
    return any(action.score >= 10.0 for action in action_scores)


def _major_setup_lock_in_active(
    *,
    own_active: PokemonState,
    own_boosts: Dict[str, int],
    action_scores: Sequence[ScoredAction],
    incoming_threat_pct: Optional[float],
) -> bool:
    if not _has_major_setup(own_boosts):
        return False
    if not _has_meaningful_active_path(action_scores):
        return False
    own_hp_pct = 100.0 * (own_active.hp_fraction if own_active.hp_fraction is not None else 1.0)
    if incoming_threat_pct is not None and incoming_threat_pct >= own_hp_pct:
        return False
    return True


def _switch_boost_multiplier(boost_total: int) -> float:
    if boost_total <= 0:
        return 1.0
    return max(0.05, 0.45 / (1.0 + 0.35 * boost_total))


def _voluntary_switch_scores(
    *,
    context: Dict[str, Any],
    request: Dict[str, Any],
    own_active: PokemonState,
    opponent: PokemonState,
    public_lines: Sequence[str],
    existing_scores: Sequence[ScoredAction],
    best_damage_score: float,
    own_boosts: Dict[str, int],
    active_has_reliable_ko: bool,
    force_switch_only: bool,
    active_sleep_source: Optional[str],
    major_setup_locked: bool,
    project_root: Path,
    calc_timeout_seconds: int,
) -> List[ScoredAction]:
    switches = _available_switches(request)
    if not switches:
        return []
    if active_has_reliable_ko and not force_switch_only:
        return []
    best_existing = max((action.score for action in existing_scores), default=0.0)
    active_immobile = own_active.status == "frz" or (
        own_active.status == "slp" and active_sleep_source != SLEEP_SOURCE_REST
    )
    if major_setup_locked and not force_switch_only and not active_immobile:
        return []
    if (
        not force_switch_only
        and not active_immobile
        and (best_existing >= 45.0 or (best_existing >= 35.0 and best_damage_score >= 20.0))
    ):
        return []

    opponent_moves = _revealed_opponent_moves(context, public_lines)[:4]
    if not force_switch_only and not active_immobile and not opponent_moves and best_existing >= 20.0:
        return []
    if (
        not force_switch_only
        and not active_immobile
        and _own_active_move_count_since_switch(context, public_lines, own_active.ident) < 2
        and best_existing > 0
    ):
        return []
    after_forced_switch = _active_entered_after_own_faint(context, public_lines, own_active.ident)
    boost_total = _positive_boost_total(own_boosts)
    boosted_active = boost_total > 0
    scores: List[ScoredAction] = []
    faster_ko_scores: List[ScoredAction] = []
    for choice, label, pokemon in switches:
        evaluation = _switch_candidate_evaluation(
            pokemon=pokemon,
            opponent=opponent,
            opponent_moves=opponent_moves,
            project_root=project_root,
            calc_timeout_seconds=calc_timeout_seconds,
        )
        if evaluation is None:
            continue
        raw_score = evaluation.score
        if active_immobile:
            required_score = max(10.0, best_existing * 0.25)
            if raw_score < required_score:
                continue
            switch_score = raw_score * 1.25 + 25.0
        elif force_switch_only:
            if raw_score <= 0:
                continue
            switch_score = raw_score + 35.0
        elif after_forced_switch:
            if best_existing >= 10.0 or raw_score < 80.0:
                continue
            switch_score = raw_score * 0.35
        else:
            required_score = 30.0 if best_existing < 10.0 else max(45.0, best_existing + 12.0)
            if boosted_active:
                required_score = max(
                    required_score + 15.0 + (8.0 * boost_total),
                    best_existing + 20.0 + (6.0 * boost_total),
                    55.0 + (5.0 * boost_total),
                )
            if raw_score < required_score:
                continue
            switch_score = raw_score * 0.65
        if boosted_active and not active_immobile:
            switch_score *= _switch_boost_multiplier(boost_total)
        if not active_immobile and (own_active.hp_fraction or 1.0) <= 0.20 and switch_score < best_existing + 25.0:
            continue
        if switch_score > 0:
            if active_immobile:
                reason = f"{own_active.status} pivot; {evaluation.reason}"
            elif force_switch_only:
                reason = f"emergency pivot; {evaluation.reason}"
            else:
                reason = (
                    f"defensive pivot despite +{boost_total} active boosts; {evaluation.reason}"
                    if boosted_active
                    else f"defensive pivot; {evaluation.reason}"
                )
            action = _scored_action(choice, label, switch_score, reason)
            if evaluation.faster and evaluation.reliable_ko and not active_has_reliable_ko:
                faster_ko_scores.append(_scored_action(choice, label, switch_score + 140.0, f"faster reliable KO pivot; {reason}"))
            else:
                scores.append(action)
    if faster_ko_scores:
        return faster_ko_scores
    return scores


def _least_bad_switch_scores(
    *,
    request: Dict[str, Any],
    opponent: PokemonState,
    opponent_moves: Sequence[str],
    project_root: Path,
    calc_timeout_seconds: int,
    reason_prefix: str,
) -> List[ScoredAction]:
    switches = _available_switches(request)
    if not switches:
        return []
    has_mobile_candidate = any(
        _condition_status(pokemon.get("condition")) not in {"slp", "frz"}
        for _, _, pokemon in switches
    )
    actions: List[ScoredAction] = []
    for choice, label, pokemon in switches:
        hp_fraction = _condition_hp_fraction(pokemon.get("condition")) or 0.0
        if hp_fraction <= 0.0:
            continue
        status = _condition_status(pokemon.get("condition"))
        if status in {"slp", "frz"} and has_mobile_candidate:
            continue
        candidate = _pokemon_state_from_side_slot(pokemon)
        candidate_hp_pct = hp_fraction * 100.0
        score = max(1.0, candidate_hp_pct * 0.20)
        reason_parts = [reason_prefix, f"bench HP {candidate_hp_pct:.0f}%"]
        if status in {"slp", "frz"}:
            score *= 0.10
            reason_parts.append(f"only immobile bench option ({status})")
        elif status in MAJOR_STATUSES:
            score *= 0.75
            reason_parts.append(f"statused {status}")

        candidate_speed = _effective_gen1_speed(candidate, project_root)
        opponent_speed = _effective_gen1_speed(opponent, project_root)
        if candidate_speed is not None and opponent_speed is not None:
            if candidate_speed > opponent_speed:
                score += 18.0
                reason_parts.append("faster")
            else:
                score *= 0.70
                reason_parts.append("slower")

        incoming = _worst_incoming_damage_pct(
            opponent=opponent,
            defender=candidate,
            move_names=opponent_moves,
            project_root=project_root,
            calc_timeout_seconds=calc_timeout_seconds,
        )
        if incoming is not None:
            score += max(0.0, 70.0 - incoming) * 0.25
            reason_parts.append(f"worst revealed incoming {incoming:.1f}%")
            if incoming >= candidate_hp_pct:
                score *= 0.25
                reason_parts.append("known KO risk")
            elif incoming >= candidate_hp_pct * 0.75:
                score *= 0.45
                reason_parts.append("heavy revealed threat")

        outgoing = _best_outgoing_damage_range(
            attacker=candidate,
            defender=opponent,
            move_names=_known_pokemon_moves(pokemon),
            project_root=project_root,
            calc_timeout_seconds=calc_timeout_seconds,
        )
        if outgoing is not None:
            min_outgoing, max_outgoing = outgoing
            target_hp_pct = 100.0 * (opponent.hp_fraction if opponent.hp_fraction is not None else 1.0)
            score += min(100.0, max_outgoing) * 0.25
            reason_parts.append(f"best outgoing {min_outgoing:.1f}-{max_outgoing:.1f}%")
            if min_outgoing >= target_hp_pct:
                score += 35.0
                reason_parts.append("reliable KO")

        actions.append(_scored_action(choice, label, max(0.01, score), ", ".join(reason_parts)))
    return actions


def _switch_candidate_evaluation(
    *,
    pokemon: Dict[str, Any],
    opponent: PokemonState,
    opponent_moves: Sequence[str],
    project_root: Path,
    calc_timeout_seconds: int,
) -> Optional[SwitchCandidateEvaluation]:
    hp_fraction = _condition_hp_fraction(pokemon.get("condition")) or 0.0
    if hp_fraction <= 0.0:
        return None
    status = _condition_status(pokemon.get("condition"))
    if status in {"slp", "frz"}:
        return None
    candidate = _pokemon_state_from_side_slot(pokemon)
    candidate_hp_pct = hp_fraction * 100.0
    candidate_speed = _effective_gen1_speed(candidate, project_root)
    opponent_speed = _effective_gen1_speed(opponent, project_root)
    faster = (
        None
        if candidate_speed is None or opponent_speed is None
        else candidate_speed > opponent_speed
    )
    score = hp_fraction * 25.0
    reason_parts = [f"bench HP {candidate_hp_pct:.0f}%"]
    incoming = _worst_incoming_damage_pct(
        opponent=opponent,
        defender=candidate,
        move_names=opponent_moves,
        project_root=project_root,
        calc_timeout_seconds=calc_timeout_seconds,
    )
    if incoming is not None and faster is False and incoming >= candidate_hp_pct:
        return None
    if incoming is not None:
        score += max(0.0, 90.0 - incoming) * 0.55
        reason_parts.append(f"worst revealed incoming {incoming:.1f}%")
        if incoming >= candidate_hp_pct * 0.75:
            score *= 0.20
            reason_parts.append("weak to revealed threat")
        elif incoming >= candidate_hp_pct * 0.50:
            score *= 0.55
            reason_parts.append("pressured by revealed threat")
    else:
        score += hp_fraction * 8.0
    outgoing = _best_outgoing_damage_range(
        attacker=candidate,
        defender=opponent,
        move_names=_known_pokemon_moves(pokemon),
        project_root=project_root,
        calc_timeout_seconds=calc_timeout_seconds,
    )
    reliable_ko = False
    if outgoing is not None:
        min_outgoing, max_outgoing = outgoing
        target_hp_pct = 100.0 * (opponent.hp_fraction if opponent.hp_fraction is not None else 1.0)
        reliable_ko = min_outgoing >= target_hp_pct
        score += min(100.0, max_outgoing) * 0.45
        reason_parts.append(f"best outgoing {min_outgoing:.1f}-{max_outgoing:.1f}%")
        if reliable_ko:
            score += 60.0
            reason_parts.append("reliable KO")
    if faster is True:
        score += 45.0
        reason_parts.append("faster")
    elif faster is False:
        reason_parts.append("slower")
    if status in MAJOR_STATUSES:
        score *= 0.80
        reason_parts.append(f"statused {status}")
    return SwitchCandidateEvaluation(
        score=max(0.0, score),
        reason=", ".join(reason_parts),
        faster=faster,
        reliable_ko=reliable_ko,
    )


def _switch_candidate_score(
    *,
    pokemon: Dict[str, Any],
    opponent: PokemonState,
    opponent_moves: Sequence[str],
    project_root: Path,
    calc_timeout_seconds: int,
) -> float:
    evaluation = _switch_candidate_evaluation(
        pokemon=pokemon,
        opponent=opponent,
        opponent_moves=opponent_moves,
        project_root=project_root,
        calc_timeout_seconds=calc_timeout_seconds,
    )
    return evaluation.score if evaluation is not None else 0.0


def _worst_incoming_damage_pct(
    *,
    opponent: PokemonState,
    defender: PokemonState,
    move_names: Sequence[str],
    project_root: Path,
    calc_timeout_seconds: int,
) -> Optional[float]:
    if not move_names:
        return None
    requests = [
        {
            "schema_version": CALC_REQUEST_SCHEMA_VERSION,
            "generation": 1,
            "attacker": opponent.calc_ref,
            "defender": defender.calc_ref,
            "move": {"name": move_name},
            "field": {},
        }
        for move_name in move_names
    ]
    try:
        response = run_damage_calc_batch(
            {"schema_version": CALC_BATCH_REQUEST_SCHEMA_VERSION, "requests": requests},
            project_root=project_root,
            timeout_seconds=calc_timeout_seconds,
        )
    except ConfigError:
        return None
    worst = 0.0
    for result in response.get("results", []):
        if not isinstance(result, dict) or result.get("status") != "ok":
            continue
        payload = result.get("result")
        range_percent = payload.get("range_percent") if isinstance(payload, dict) else None
        if isinstance(range_percent, dict):
            worst = max(worst, _float_value(range_percent.get("max")) or 0.0)
    return worst if worst > 0 else None


def _best_outgoing_damage_pct(
    *,
    attacker: PokemonState,
    defender: PokemonState,
    move_names: Sequence[str],
    project_root: Path,
    calc_timeout_seconds: int,
) -> Optional[float]:
    best_range = _best_outgoing_damage_range(
        attacker=attacker,
        defender=defender,
        move_names=move_names,
        project_root=project_root,
        calc_timeout_seconds=calc_timeout_seconds,
    )
    return best_range[1] if best_range is not None else None


def _best_outgoing_damage_range(
    *,
    attacker: PokemonState,
    defender: PokemonState,
    move_names: Sequence[str],
    project_root: Path,
    calc_timeout_seconds: int,
) -> Optional[tuple[float, float]]:
    if not move_names:
        return None
    requests = [
        {
            "schema_version": CALC_REQUEST_SCHEMA_VERSION,
            "generation": 1,
            "attacker": attacker.calc_ref,
            "defender": defender.calc_ref,
            "move": {"name": move_name},
            "field": {},
        }
        for move_name in move_names[:4]
    ]
    try:
        response = run_damage_calc_batch(
            {"schema_version": CALC_BATCH_REQUEST_SCHEMA_VERSION, "requests": requests},
            project_root=project_root,
            timeout_seconds=calc_timeout_seconds,
        )
    except ConfigError:
        return None
    best_min = 0.0
    best_max = 0.0
    for result in response.get("results", []):
        if not isinstance(result, dict) or result.get("status") != "ok":
            continue
        payload = result.get("result")
        range_percent = payload.get("range_percent") if isinstance(payload, dict) else None
        if isinstance(range_percent, dict):
            max_pct = _float_value(range_percent.get("max")) or 0.0
            if max_pct > best_max:
                best_max = max_pct
                best_min = _float_value(range_percent.get("min")) or 0.0
    return (best_min, best_max) if best_max > 0 else None


def _known_pokemon_moves(pokemon: Dict[str, Any]) -> List[str]:
    moves = pokemon.get("moves")
    if not isinstance(moves, list):
        return []
    return [str(move).strip() for move in moves if isinstance(move, str) and move.strip()]


def _available_switches(request: Dict[str, Any]) -> List[tuple[str, str, Dict[str, Any]]]:
    active = request.get("active")
    if (
        isinstance(active, list)
        and len(active) == 1
        and isinstance(active[0], dict)
        and active[0].get("trapped") is True
    ):
        return []
    side = request.get("side")
    pokemon = side.get("pokemon") if isinstance(side, dict) else None
    if not isinstance(pokemon, list):
        return []
    switches: List[tuple[str, str, Dict[str, Any]]] = []
    for index, candidate in enumerate(pokemon, start=1):
        if not isinstance(candidate, dict) or candidate.get("active"):
            continue
        condition = str(candidate.get("condition") or "")
        if "fnt" in condition:
            continue
        switches.append(
            (
                f"switch {index}",
                f'Switch to "{_pokemon_species(candidate) or f"slot {index}"}"',
                candidate,
            )
        )
    return switches


def _remaining_pokemon_advantage(context: Dict[str, Any], public_lines: Sequence[str]) -> int:
    own_remaining = _remaining_own_pokemon(context)
    opponent_remaining = _remaining_opponent_pokemon(context, public_lines)
    if own_remaining is None or opponent_remaining is None:
        return 0
    return own_remaining - opponent_remaining


def _remaining_own_pokemon(context: Dict[str, Any]) -> Optional[int]:
    pokemon = _side_pokemon(context)
    if not pokemon:
        return None
    remaining = 0
    for candidate in pokemon:
        if not isinstance(candidate, dict):
            continue
        if "fnt" not in str(candidate.get("condition") or ""):
            remaining += 1
    return remaining


def _remaining_opponent_pokemon(context: Dict[str, Any], public_lines: Sequence[str]) -> Optional[int]:
    team_size = _side_team_size(context)
    if team_size is None:
        return None
    opponent_slot = _opponent_slot(_effective_player_slot(context))
    fainted = {
        _identity_key(parts[2])
        for line in public_lines
        if isinstance(line, str)
        for parts in [line.split("|")]
        if len(parts) >= 3 and parts[1] == "faint" and _ident_matches_slot(parts[2], opponent_slot)
    }
    return max(0, team_size - len(fainted))


def _side_team_size(context: Dict[str, Any]) -> Optional[int]:
    pokemon = _side_pokemon(context)
    return len(pokemon) if pokemon else None


def _side_pokemon(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    side = _context_side(context)
    pokemon = side.get("pokemon") if isinstance(side, dict) else None
    if not isinstance(pokemon, list):
        return []
    return [candidate for candidate in pokemon if isinstance(candidate, dict)]


def _choose_weighted(actions: Sequence[ScoredAction], rng: random.Random) -> str:
    total_weight = sum(max(0.0, action.weight) for action in actions)
    if total_weight <= 0:
        return actions[0].choice
    roll = rng.random() * total_weight
    cumulative = 0.0
    for action in actions:
        cumulative += max(0.0, action.weight)
        if roll <= cumulative:
            return action.choice
    return actions[-1].choice


def _select_action(
    actions: Sequence[ScoredAction],
    rng: random.Random,
    selection_strategy: str = DEFAULT_CUSTOM_BOT_SELECTION_STRATEGY,
) -> tuple[str, List[ScoredAction]]:
    strategy = _normalize_selection_strategy(selection_strategy)
    weighted_actions = _actions_with_selection_strategy(actions, strategy)
    if not weighted_actions:
        raise ConfigError("Cannot select from an empty custom-bot action list.")
    if strategy == SELECTION_ARGMAX:
        best_score = max(action.score for action in weighted_actions)
        tied = [action for action in weighted_actions if action.score == best_score]
        return rng.choice(tied).choice, weighted_actions
    return _choose_weighted(weighted_actions, rng), weighted_actions


def _actions_with_selection_strategy(
    actions: Sequence[ScoredAction],
    selection_strategy: str = DEFAULT_CUSTOM_BOT_SELECTION_STRATEGY,
) -> List[ScoredAction]:
    strategy = _normalize_selection_strategy(selection_strategy)
    if strategy == SELECTION_ARGMAX:
        best_score = max((action.score for action in actions), default=0.0)
        return [
            replace(action, weight=1.0 if action.score == best_score and best_score > 0 else 0.0)
            for action in actions
        ]
    return [
        replace(action, weight=_selection_weight(action.score, strategy))
        for action in actions
    ]


def _selection_weight(score: float, selection_strategy: str) -> float:
    safe_score = max(0.0, float(score))
    if selection_strategy == SELECTION_WEIGHTED_LINEAR:
        return safe_score
    if selection_strategy == SELECTION_WEIGHTED_CUBE:
        return safe_score ** 3
    return safe_score * safe_score


def _normalize_selection_strategy(selection_strategy: str) -> str:
    strategy = str(selection_strategy or DEFAULT_CUSTOM_BOT_SELECTION_STRATEGY).strip().lower()
    if strategy not in CUSTOM_BOT_SELECTION_STRATEGIES:
        allowed = ", ".join(CUSTOM_BOT_SELECTION_STRATEGIES)
        raise ConfigError(f"Unknown custom bot selection strategy {selection_strategy!r}. Available: {allowed}.")
    return strategy


def _selection_pool(actions: Sequence[ScoredAction]) -> List[ScoredAction]:
    if len(actions) <= 1:
        return list(actions)
    best_score = max(action.score for action in actions)
    if best_score < 10.0:
        return list(actions)
    minimum_competitive_score = max(10.0, best_score * 0.55)
    pool = [action for action in actions if action.score >= minimum_competitive_score]
    return pool or list(actions)


def _scored_action(choice: str, label: str, score: float, reason: str) -> ScoredAction:
    safe_score = max(0.0, float(score))
    return ScoredAction(
        choice=choice,
        label=label,
        score=safe_score,
        weight=safe_score * safe_score,
        reason=reason,
    )


def _notes(
    decision: str,
    actions: Sequence[ScoredAction],
    fallback_reason: Optional[str],
    warnings: Optional[Sequence[str]] = None,
    selection_strategy: str = DEFAULT_CUSTOM_BOT_SELECTION_STRATEGY,
) -> str:
    if fallback_reason:
        return f"custom-bot fallback: {fallback_reason}; selected {decision}."
    total_weight = sum(action.weight for action in actions)
    ranked = sorted(actions, key=lambda item: item.score, reverse=True)[:6]
    parts = []
    for action in ranked:
        probability = (action.weight / total_weight * 100.0) if total_weight > 0 else 0.0
        parts.append(
            f"{action.choice} {action.label}: score={action.score:.1f}, "
            f"p={probability:.0f}%, {action.reason}"
        )
    warning_text = " ".join(warnings or [])
    strategy = _normalize_selection_strategy(selection_strategy)
    score_text = (
        f"custom-bot gen1randombattle weighted scores (selection={strategy}); "
        f"selected {decision}. "
        + "; ".join(parts)
    )
    return f"{warning_text} {score_text}".strip()


def _active_side_pokemon(side: Optional[Dict[str, Any]]) -> Optional[PokemonState]:
    if not isinstance(side, dict):
        return None
    pokemon = side.get("pokemon")
    if not isinstance(pokemon, list):
        return None
    for candidate in pokemon:
        if isinstance(candidate, dict) and candidate.get("active"):
            return _pokemon_state_from_side_slot(candidate)
    return None


def _pokemon_state_from_side_slot(pokemon: Dict[str, Any]) -> PokemonState:
    species = _pokemon_species(pokemon) or "Mew"
    options: Dict[str, Any] = {}
    level = _pokemon_level(pokemon)
    if level is not None:
        options["level"] = level
    stats = pokemon.get("stats")
    if isinstance(stats, dict) and stats:
        options["stats"] = stats
    item = pokemon.get("item")
    if isinstance(item, str) and item.strip():
        options["item"] = item.strip()
    ability = pokemon.get("baseAbility")
    if isinstance(ability, str) and ability.strip():
        options["ability"] = ability.strip()
    calc_ref: Dict[str, Any] = {"species": species}
    if options:
        calc_ref["options"] = options
    condition = pokemon.get("condition")
    return PokemonState(
        species=species,
        ident=str(pokemon.get("ident")) if pokemon.get("ident") else None,
        calc_ref=calc_ref,
        hp_fraction=_condition_hp_fraction(condition),
        status=_condition_status(condition),
        volatile_statuses=_volatile_statuses_from_value(
            pokemon.get("volatile_status") if "volatile_status" in pokemon else pokemon.get("volatiles")
        ),
    )


def _with_volatile_statuses(
    pokemon: PokemonState,
    volatile_statuses: Sequence[str],
) -> PokemonState:
    normalized = tuple(dict.fromkeys(_normalize_move_id(status) for status in volatile_statuses if status))
    if normalized == pokemon.volatile_statuses:
        return pokemon
    return PokemonState(
        species=pokemon.species,
        ident=pokemon.ident,
        calc_ref=pokemon.calc_ref,
        hp_fraction=pokemon.hp_fraction,
        status=pokemon.status,
        volatile_statuses=normalized,
    )


def _active_opponent_from_public_history(
    context: Dict[str, Any],
    public_lines: Sequence[str],
) -> Optional[PokemonState]:
    opponent_prefix = _opponent_slot(_effective_player_slot(context))
    active: Optional[PokemonState] = None
    for line in public_lines:
        if not isinstance(line, str):
            continue
        parts = line.split("|")
        if (
            len(parts) >= 4
            and parts[1] in {"switch", "drag", "replace"}
            and _ident_matches_slot(parts[2], opponent_prefix)
        ):
            details = parts[3] if len(parts) > 3 else ""
            condition = parts[4] if len(parts) > 4 else None
            species = _species_from_details(details) or _species_from_ident(parts[2])
            if species:
                options: Dict[str, Any] = {}
                level = _level_from_details(details)
                if level is not None:
                    options["level"] = level
                calc_ref: Dict[str, Any] = {"species": species}
                if options:
                    calc_ref["options"] = options
                active = PokemonState(
                    species=species,
                    ident=parts[2],
                    calc_ref=calc_ref,
                    hp_fraction=_condition_hp_fraction(condition),
                    status=_condition_status(condition),
                    volatile_statuses=(),
                )
            continue
        if active is None or len(parts) < 4:
            continue
        if parts[1] in {"-damage", "-heal"} and _same_ident(parts[2], active.ident):
            condition = parts[3] if len(parts) > 3 else None
            active = PokemonState(
                species=active.species,
                ident=active.ident,
                calc_ref=active.calc_ref,
                hp_fraction=_condition_hp_fraction(condition),
                status=_condition_status(condition) or active.status,
                volatile_statuses=active.volatile_statuses,
            )
        elif parts[1] == "-status" and _same_ident(parts[2], active.ident):
            active = PokemonState(
                active.species,
                active.ident,
                active.calc_ref,
                active.hp_fraction,
                parts[3],
                active.volatile_statuses,
            )
        elif parts[1] == "-curestatus" and _same_ident(parts[2], active.ident):
            active = PokemonState(
                active.species,
                active.ident,
                active.calc_ref,
                active.hp_fraction,
                None,
                active.volatile_statuses,
            )
        elif parts[1] == "faint" and _same_ident(parts[2], active.ident):
            active = PokemonState(
                active.species,
                active.ident,
                active.calc_ref,
                0.0,
                active.status,
                active.volatile_statuses,
            )
        elif parts[1] == "-start" and _same_ident(parts[2], active.ident):
            active = PokemonState(
                active.species,
                active.ident,
                active.calc_ref,
                active.hp_fraction,
                active.status,
                _add_volatile_status(active.volatile_statuses, parts[3]),
            )
        elif parts[1] == "-end" and _same_ident(parts[2], active.ident):
            active = PokemonState(
                active.species,
                active.ident,
                active.calc_ref,
                active.hp_fraction,
                active.status,
                _remove_volatile_status(active.volatile_statuses, parts[3]),
            )
    return active


def _public_lines(context: Dict[str, Any], capture_payload: Optional[Dict[str, Any]]) -> List[str]:
    if isinstance(capture_payload, dict) and isinstance(capture_payload.get("events"), list):
        lines: List[str] = []
        for event in capture_payload["events"]:
            if not isinstance(event, dict):
                continue
            payload = event.get("payload")
            event_lines = payload.get("lines") if isinstance(payload, dict) else None
            if isinstance(event_lines, list):
                lines.extend(line for line in event_lines if isinstance(line, str))
        if lines:
            return lines
    recent = context.get("recent_public_events")
    return [line for line in recent if isinstance(line, str)] if isinstance(recent, list) else []


def _active_sleep_source(
    context: Dict[str, Any],
    public_lines: Sequence[str],
    own_active: PokemonState,
) -> Optional[str]:
    if own_active.status != "slp":
        return None
    player_slot = _effective_player_slot(context)
    opponent_slot = _opponent_slot(player_slot)
    source: Optional[str] = None
    last_move: Optional[tuple[str, str, str]] = None
    saw_recent_own_rest = False
    for line in public_lines:
        parts = line.split("|") if isinstance(line, str) else []
        if len(parts) >= 4 and parts[1] in {"switch", "drag", "replace"} and _same_ident(parts[2], own_active.ident):
            source = None
            last_move = None
            saw_recent_own_rest = False
            continue
        if len(parts) >= 4 and parts[1] == "move":
            actor = parts[2]
            move_name = parts[3]
            target = parts[4] if len(parts) > 4 else ""
            last_move = (actor, move_name, target)
            if _same_ident(actor, own_active.ident) and _normalize_move_id(move_name) == "rest":
                saw_recent_own_rest = True
            continue
        if len(parts) >= 4 and parts[1] == "-status" and parts[3] == "slp" and _same_ident(parts[2], own_active.ident):
            if _status_line_has_rest_source(parts):
                source = SLEEP_SOURCE_REST
            elif (
                last_move is not None
                and _same_ident(last_move[0], own_active.ident)
                and _normalize_move_id(last_move[1]) == "rest"
            ):
                source = SLEEP_SOURCE_REST
            elif (
                last_move is not None
                and _ident_matches_slot(last_move[0], opponent_slot)
                and _same_ident(last_move[2], parts[2])
                and _normalize_move_id(last_move[1]) in SLEEP_MOVES
            ):
                source = SLEEP_SOURCE_OPPONENT
            else:
                source = SLEEP_SOURCE_UNKNOWN
            continue
        if len(parts) >= 4 and parts[1] == "-curestatus" and parts[3] == "slp" and _same_ident(parts[2], own_active.ident):
            source = None
            saw_recent_own_rest = False
            continue
        if len(parts) >= 3 and parts[1] == "faint" and _same_ident(parts[2], own_active.ident):
            source = None
            saw_recent_own_rest = False
    if source is not None:
        return source
    if saw_recent_own_rest:
        return SLEEP_SOURCE_REST
    return SLEEP_SOURCE_UNKNOWN


def _status_line_has_rest_source(parts: Sequence[str]) -> bool:
    return any("move: rest" in part.lower() for part in parts[4:])


def _sleep_clause_active(context: Dict[str, Any], public_lines: Sequence[str]) -> bool:
    player_slot = _effective_player_slot(context)
    opponent_slot = _opponent_slot(player_slot)
    asleep: Dict[str, bool] = {}
    last_move: Optional[tuple[str, str, str]] = None
    for line in public_lines:
        parts = line.split("|") if isinstance(line, str) else []
        if len(parts) >= 4 and parts[1] == "move":
            actor = parts[2]
            move_name = parts[3]
            target = parts[4] if len(parts) > 4 else ""
            last_move = (actor, move_name, target)
            continue
        if (
            len(parts) >= 4
            and parts[1] == "-status"
            and parts[3] == "slp"
            and _ident_matches_slot(parts[2], opponent_slot)
        ):
            if (
                last_move is not None
                and _ident_matches_slot(last_move[0], player_slot)
                and _same_ident(last_move[2], parts[2])
                and _normalize_move_id(last_move[1]) in SLEEP_MOVES
            ):
                asleep[_identity_key(parts[2])] = True
            continue
        if (
            len(parts) >= 3
            and parts[1] == "faint"
            and _ident_matches_slot(parts[2], opponent_slot)
        ):
            asleep.pop(_identity_key(parts[2]), None)
        if len(parts) >= 3 and parts[1] == "-curestatus" and _ident_matches_slot(parts[2], opponent_slot):
            cured_status = parts[3] if len(parts) > 3 else ""
            if cured_status == "slp":
                asleep.pop(_identity_key(parts[2]), None)
    return any(asleep.values())


def _boosts_for_active(
    context: Dict[str, Any],
    public_lines: Sequence[str],
    own_ident: Optional[str],
) -> Dict[str, int]:
    _ = context
    boosts: Dict[str, int] = {}
    if not own_ident:
        return boosts
    for line in public_lines:
        parts = line.split("|") if isinstance(line, str) else []
        if len(parts) >= 4 and parts[1] in {"switch", "drag", "replace"} and _same_ident(parts[2], own_ident):
            boosts = {}
            continue
        if len(parts) >= 5 and parts[1] == "-boost" and _same_ident(parts[2], own_ident):
            stat = parts[3]
            boosts[stat] = min(BOOST_CAP, boosts.get(stat, 0) + _int_value(parts[4], 0))
            continue
        if len(parts) >= 5 and parts[1] == "-unboost" and _same_ident(parts[2], own_ident):
            stat = parts[3]
            boosts[stat] = max(-BOOST_CAP, boosts.get(stat, 0) - _int_value(parts[4], 0))
            continue
        if len(parts) >= 3 and parts[1] in {"-clearboost", "-clearallboost"} and _same_ident(parts[2], own_ident):
            boosts = {}
    return boosts


def _volatile_statuses_for_active(
    public_lines: Sequence[str],
    own_ident: Optional[str],
    initial: Sequence[str] = (),
) -> tuple[str, ...]:
    volatile_statuses = tuple(_normalize_move_id(status) for status in initial if status)
    if not own_ident:
        return volatile_statuses
    for line in public_lines:
        parts = line.split("|") if isinstance(line, str) else []
        if len(parts) >= 4 and parts[1] in {"switch", "drag", "replace"} and _same_ident(parts[2], own_ident):
            volatile_statuses = ()
            continue
        if len(parts) >= 4 and parts[1] == "-start" and _same_ident(parts[2], own_ident):
            volatile_statuses = _add_volatile_status(volatile_statuses, parts[3])
            continue
        if len(parts) >= 4 and parts[1] == "-end" and _same_ident(parts[2], own_ident):
            volatile_statuses = _remove_volatile_status(volatile_statuses, parts[3])
    return volatile_statuses


def _revealed_opponent_moves(context: Dict[str, Any], public_lines: Sequence[str]) -> List[str]:
    opponent_slot = _opponent_slot(_effective_player_slot(context))
    moves: List[str] = []
    seen: set[str] = set()
    for line in public_lines:
        parts = line.split("|") if isinstance(line, str) else []
        if len(parts) >= 4 and parts[1] == "move" and _ident_matches_slot(parts[2], opponent_slot):
            move_name = parts[3].strip()
            key = _normalize_move_id(move_name)
            if move_name and key not in seen:
                seen.add(key)
                moves.append(move_name)
    return moves


def _counter_revealed(opponent_moves: Sequence[str]) -> bool:
    return any(_normalize_move_id(move) == "counter" for move in opponent_moves)


def _active_entered_after_own_faint(
    context: Dict[str, Any],
    public_lines: Sequence[str],
    own_ident: Optional[str],
) -> bool:
    if not own_ident:
        return False
    player_slot = _effective_player_slot(context)
    last_own_event: Optional[str] = None
    active_switch_after_faint = False
    active_switch_seen = False
    for line in public_lines:
        parts = line.split("|") if isinstance(line, str) else []
        if len(parts) < 3:
            continue
        event_type = parts[1]
        actor = parts[2]
        if not _ident_matches_slot(actor, player_slot):
            continue
        if event_type in {"switch", "drag", "replace"}:
            if _same_ident(actor, own_ident):
                active_switch_after_faint = last_own_event == "faint"
                active_switch_seen = True
            last_own_event = "switch"
            continue
        if event_type == "faint":
            last_own_event = "faint"
            continue
        if event_type == "move":
            if active_switch_seen and _same_ident(actor, own_ident):
                active_switch_after_faint = False
            last_own_event = "move"
    return active_switch_after_faint


def _own_active_move_count_since_switch(
    context: Dict[str, Any],
    public_lines: Sequence[str],
    own_ident: Optional[str],
) -> int:
    if not own_ident:
        return 999
    player_slot = _effective_player_slot(context)
    active_seen = False
    move_count = 999
    for line in public_lines:
        parts = line.split("|") if isinstance(line, str) else []
        if len(parts) < 3:
            continue
        event_type = parts[1]
        actor = parts[2]
        if not _ident_matches_slot(actor, player_slot):
            continue
        if event_type in {"switch", "drag", "replace"}:
            active_seen = _same_ident(actor, own_ident)
            move_count = 0 if active_seen else 999
            continue
        if not active_seen or not _same_ident(actor, own_ident):
            continue
        if event_type == "move":
            move_count += 1
        elif event_type == "faint":
            active_seen = False
            move_count = 999
    return move_count


def _heuristic_move_metadata(move_name: str, request_move: Dict[str, Any]) -> Dict[str, Any]:
    category = request_move.get("category")
    base_power = request_move.get("basePower")
    return {
        "requested_name": move_name,
        "id": str(request_move.get("id") or _normalize_move_id(move_name)),
        "name": move_name,
        "category": category if category in {"Physical", "Special", "Status"} else "Physical",
        "base_power": base_power if isinstance(base_power, int) else 1,
        "type": request_move.get("type") if isinstance(request_move.get("type"), str) else None,
        "accuracy": 100,
        "flags": {},
        "boosts": {},
        "self": {},
        "secondary": {},
        "high_crit": False,
        "recharge": _normalize_move_id(move_name) == "hyperbeam",
        "charge": _normalize_move_id(move_name) in CHARGE_MOVE_IDS,
    }


def _is_damaging_metadata(metadata: Dict[str, Any]) -> bool:
    return metadata.get("category") != "Status" and (_int_value(metadata.get("base_power"), 0) > 0)


def _accuracy_factor(metadata: Dict[str, Any]) -> float:
    accuracy = metadata.get("accuracy")
    if accuracy is True:
        return 1.0
    if isinstance(accuracy, bool):
        return 1.0
    if isinstance(accuracy, (int, float)):
        return max(0.0, min(1.0, float(accuracy) / 100.0))
    return 1.0


def _known_speed_stat(pokemon: PokemonState) -> Optional[int]:
    options = pokemon.calc_ref.get("options") if isinstance(pokemon.calc_ref, dict) else None
    stats = options.get("stats") if isinstance(options, dict) else None
    if not isinstance(stats, dict):
        return None
    speed = _int_value(stats.get("spe"), 0)
    return speed if speed > 0 else None


def _estimated_gen1_speed(pokemon: PokemonState, project_root: Path) -> Optional[int]:
    known_speed = _known_speed_stat(pokemon)
    if known_speed is not None:
        return known_speed
    base_speed = _base_speed_for_species(pokemon.species, project_root)
    if base_speed is None:
        return None
    level = _calc_ref_level(pokemon.calc_ref) or 100
    return int(((((base_speed + 15) * 2 + 63) * level) // 100) + 5)


def _effective_gen1_speed(
    pokemon: PokemonState,
    project_root: Path,
    *,
    speed_boost: int = 0,
) -> Optional[float]:
    speed = _estimated_gen1_speed(pokemon, project_root)
    if speed is None:
        return None
    return _effective_speed_value(speed, status=pokemon.status, speed_boost=speed_boost)


def _effective_speed_value(speed: float, *, status: Optional[str], speed_boost: int = 0) -> float:
    modified = float(speed) * _boost_multiplier(speed_boost)
    if status == "par" and speed_boost <= 0:
        modified /= 4.0
    return modified


def _boost_multiplier(boost: int) -> float:
    boost = max(-BOOST_CAP, min(BOOST_CAP, boost))
    if boost >= 0:
        return (2.0 + boost) / 2.0
    return 2.0 / (2.0 - boost)


def _base_speed_for_species(species: str, project_root: Path) -> Optional[int]:
    try:
        root = project_root.resolve()
    except OSError:
        root = project_root
    if root not in _POKEDEX_BASE_SPEED_CACHE:
        _POKEDEX_BASE_SPEED_CACHE[root] = _load_pokedex_base_spe(root)
    return _POKEDEX_BASE_SPEED_CACHE[root].get(_normalize_move_id(species))


def _pokemon_types(pokemon: PokemonState, project_root: Path) -> List[str]:
    try:
        root = project_root.resolve()
    except OSError:
        root = project_root
    if root not in _POKEDEX_TYPES_CACHE:
        _POKEDEX_TYPES_CACHE[root] = _load_pokedex_types(root)
    return list(_POKEDEX_TYPES_CACHE[root].get(_normalize_move_id(pokemon.species), []))


def _load_pokedex_base_spe(project_root: Path) -> Dict[str, int]:
    parsed = _load_pokedex_entries(project_root)
    return {species_id: item["spe"] for species_id, item in parsed.items() if "spe" in item}


def _load_pokedex_types(project_root: Path) -> Dict[str, List[str]]:
    parsed = _load_pokedex_entries(project_root)
    return {species_id: list(item["types"]) for species_id, item in parsed.items() if "types" in item}


def _load_pokedex_entries(project_root: Path) -> Dict[str, Dict[str, Any]]:
    for relative_path in (
        Path("vendor/pokemon-showdown/dist/data/pokedex.js"),
        Path("vendor/pokemon-showdown/data/pokedex.ts"),
    ):
        path = project_root / relative_path
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        entries: Dict[str, Dict[str, Any]] = {}
        entry_pattern = re.compile(
            r"^\s{2}([a-z0-9]+):\s*\{\n(.*?)(?=^\s{2}[a-z0-9]+:\s*\{|\n\};)",
            re.MULTILINE | re.DOTALL,
        )
        for match in entry_pattern.finditer(text):
            body = match.group(2)
            types_match = re.search(r"types:\s*\[([^\]]+)\]", body)
            speed_match = re.search(r"baseStats:\s*\{[^}]*\bspe:\s*(\d+)", body)
            if types_match is None and speed_match is None:
                continue
            entry: Dict[str, Any] = {}
            if types_match is not None:
                entry["types"] = re.findall(r'"([^"]+)"', types_match.group(1))
            if speed_match is not None:
                entry["spe"] = int(speed_match.group(1))
            entries[match.group(1)] = entry
        if entries:
            return entries
    return {}


def _calc_ref_level(calc_ref: Dict[str, Any]) -> Optional[int]:
    options = calc_ref.get("options") if isinstance(calc_ref, dict) else None
    if not isinstance(options, dict):
        return None
    level = _int_value(options.get("level"), 0)
    return level if level > 0 else None


def _metadata_id(metadata: Dict[str, Any]) -> str:
    return _normalize_move_id(str(metadata.get("id") or metadata.get("name") or ""))


def _pokemon_species(pokemon: Dict[str, Any]) -> Optional[str]:
    details = pokemon.get("details")
    if isinstance(details, str):
        species = _species_from_details(details)
        if species:
            return species
    ident = pokemon.get("ident")
    if isinstance(ident, str):
        return _species_from_ident(ident)
    return None


def _pokemon_level(pokemon: Dict[str, Any]) -> Optional[int]:
    details = pokemon.get("details")
    return _level_from_details(details) if isinstance(details, str) else None


def _species_from_details(details: str) -> Optional[str]:
    value = details.split(",", 1)[0].strip()
    return value or None


def _species_from_ident(ident: str) -> Optional[str]:
    if ":" not in ident:
        return ident.strip() or None
    value = ident.split(":", 1)[1].strip()
    return value or None


def _level_from_details(details: str) -> Optional[int]:
    match = re.search(r"\bL(\d+)\b", details)
    return int(match.group(1)) if match else None


def _condition_hp_fraction(condition: Any) -> Optional[float]:
    if not isinstance(condition, str):
        return None
    if "fnt" in condition:
        return 0.0
    hp_token = condition.split(maxsplit=1)[0]
    if hp_token.endswith("%"):
        return max(0.0, min(1.0, (_float_value(hp_token[:-1]) or 0.0) / 100.0))
    if "/" not in hp_token:
        return None
    current, _, maximum = hp_token.partition("/")
    current_value = _float_value(current)
    maximum_value = _float_value(maximum)
    if current_value is None or maximum_value is None or maximum_value <= 0:
        return None
    return max(0.0, min(1.0, current_value / maximum_value))


def _condition_status(condition: Any) -> Optional[str]:
    if not isinstance(condition, str):
        return None
    tokens = condition.split()
    for token in tokens[1:]:
        if token in MAJOR_STATUSES:
            return token
    return None


def _volatile_statuses_from_value(value: Any) -> tuple[str, ...]:
    statuses: List[str] = []
    if isinstance(value, str):
        statuses = [value]
    elif isinstance(value, list):
        statuses = [item for item in value if isinstance(item, str)]
    elif isinstance(value, dict):
        statuses = [key for key in value if isinstance(key, str)]
    normalized: List[str] = []
    seen: set[str] = set()
    for status in statuses:
        key = _normalize_move_id(status)
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return tuple(normalized)


def _has_volatile_status(pokemon: PokemonState, status: str) -> bool:
    return _normalize_move_id(status) in pokemon.volatile_statuses


def _add_volatile_status(existing: Sequence[str], status: str) -> tuple[str, ...]:
    key = _normalize_move_id(status)
    if not key:
        return tuple(existing)
    values = list(existing)
    if key not in values:
        values.append(key)
    return tuple(values)


def _remove_volatile_status(existing: Sequence[str], status: str) -> tuple[str, ...]:
    key = _normalize_move_id(status)
    return tuple(value for value in existing if value != key)


def _opponent_slot(player_slot: str) -> str:
    return "p2" if player_slot.startswith("p1") else "p1"


def _effective_player_slot(context: Dict[str, Any]) -> str:
    side = _context_side(context)
    if isinstance(side, dict):
        side_id = side.get("id")
        if isinstance(side_id, str) and side_id in {"p1", "p2"}:
            return side_id
    player_slot = str(context.get("player_slot") or "p1")
    return "p2" if player_slot.startswith("p2") else "p1"


def _ident_matches_slot(ident: str, slot: str) -> bool:
    return isinstance(ident, str) and ident.startswith(slot)


def _same_ident(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False
    return _identity_key(left) == _identity_key(right)


def _identity_key(ident: str) -> str:
    value = ident.strip()
    if ":" in value:
        value = value.split(":", 1)[1].strip()
    return value.lower()


def _normalize_move_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _float_value(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _int_value(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"Failed to read {label} JSON from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigError(f"{label.capitalize()} at {path} must contain a JSON object.")
    return payload
