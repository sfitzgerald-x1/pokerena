from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote

from .agent import AgentContextCursor, BattleSession, SessionEvent, SimStreamAdapter, choose_first_legal
from .config import AgentCallableConfig, AgentLaunchConfig, ConfigError, load_agents_config, load_server_config
from .custom_bot import (
    DEFAULT_CUSTOM_BOT_SELECTION_STRATEGY,
    SELECTION_ARGMAX,
    SELECTION_WEIGHTED_CUBE,
    SELECTION_WEIGHTED_LINEAR,
    SELECTION_WEIGHTED_SQUARE,
    build_custom_bot_plan,
)
from .max_damage_bot import build_max_damage_bot_plan
from .random_bot import build_random_bot_decision


EVAL_RUN_SCHEMA_VERSION = "pokerena.eval-run.v1"
DEFAULT_EVAL_VIEWER_HOST = "127.0.0.1"
DEFAULT_EVAL_VIEWER_PORT = 8002
GEN1_RANDBAT_FORMAT = "gen1randombattle"


@dataclass(frozen=True)
class EvalBotSpec:
    bot_id: str
    label: str
    provider: str
    hook_command: str
    custom_selection_strategy: Optional[str] = None


EVAL_BOT_SPECS: Dict[str, EvalBotSpec] = {
    "custom-bot": EvalBotSpec(
        bot_id="custom-bot",
        label="Gen1CustomBot",
        provider="pokerena-custom-bot",
        hook_command="custom-bot",
        custom_selection_strategy=SELECTION_WEIGHTED_SQUARE,
    ),
    "custom-bot-argmax": EvalBotSpec(
        bot_id="custom-bot-argmax",
        label="Gen1CustomBotArgmax",
        provider="pokerena-custom-bot-argmax",
        hook_command="custom-bot",
        custom_selection_strategy=SELECTION_ARGMAX,
    ),
    "custom-bot-cube": EvalBotSpec(
        bot_id="custom-bot-cube",
        label="Gen1CustomBotCube",
        provider="pokerena-custom-bot-cube",
        hook_command="custom-bot",
        custom_selection_strategy=SELECTION_WEIGHTED_CUBE,
    ),
    "custom-bot-linear": EvalBotSpec(
        bot_id="custom-bot-linear",
        label="Gen1CustomBotLinear",
        provider="pokerena-custom-bot-linear",
        hook_command="custom-bot",
        custom_selection_strategy=SELECTION_WEIGHTED_LINEAR,
    ),
    "max-damage-bot": EvalBotSpec(
        bot_id="max-damage-bot",
        label="Gen1MaxDamageBot",
        provider="pokerena-max-damage-bot",
        hook_command="max-damage-bot",
    ),
    "random-bot": EvalBotSpec(
        bot_id="random-bot",
        label="Gen1RandomBot",
        provider="pokerena-random-bot",
        hook_command="random-bot",
    ),
}


def default_eval_runs_root(project_root: Path) -> Path:
    return project_root / ".runtime" / "eval-runs"


def eval_viewer_url(host: str = DEFAULT_EVAL_VIEWER_HOST, port: int = DEFAULT_EVAL_VIEWER_PORT) -> str:
    display_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{display_host}:{port}"


def serve_eval_viewer(
    *,
    project_root: Path,
    host: str = DEFAULT_EVAL_VIEWER_HOST,
    port: int = DEFAULT_EVAL_VIEWER_PORT,
) -> None:
    runs_root = default_eval_runs_root(project_root)
    handler = _build_eval_handler(runs_root)
    httpd = ThreadingHTTPServer((host, port), handler)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


def run_max_damage_vs_custom_eval(
    *,
    project_root: Path,
    config_path: str,
    agents_config_path: str,
    games: int,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    return run_pool_eval(
        project_root=project_root,
        config_path=config_path,
        agents_config_path=agents_config_path,
        games_per_pair=games,
        bot_ids=["max-damage-bot", "custom-bot"],
        run_id=run_id,
        default_run_prefix="max-damage-vs-custom",
    )


def run_pool_eval(
    *,
    project_root: Path,
    config_path: str,
    agents_config_path: str,
    games_per_pair: int,
    bot_ids: list[str],
    run_id: Optional[str] = None,
    default_run_prefix: str = "gen1-bot-pool",
) -> Dict[str, Any]:
    if games_per_pair <= 0:
        raise ConfigError("--games-per-pair must be greater than zero.")
    bot_specs = _resolve_bot_specs(bot_ids)
    pairings = _pairings(bot_specs)
    total_games = games_per_pair * len(pairings)
    server_config = load_server_config(config_path=config_path, project_root=project_root)
    agents = load_agents_config(config_path=agents_config_path, project_root=project_root)
    if not agents:
        raise ConfigError("At least one agent config entry is required as a template for eval contexts.")

    run_id = _safe_run_id(run_id or f"{default_run_prefix}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}")
    run_dir = default_eval_runs_root(project_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {
        "schema_version": EVAL_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "kind": "gen1-pool-round-robin",
        "status": "running",
        "format": GEN1_RANDBAT_FORMAT,
        "games_per_pair": games_per_pair,
        "games_total": total_games,
        "games_completed": 0,
        "current_game": None,
        "started_at": _timestamp(),
        "updated_at": _timestamp(),
        "completed_at": None,
        "bots": [spec.label for spec in bot_specs],
        "bot_ids": [spec.bot_id for spec in bot_specs],
        "pairings": [
            {"pair_key": _pair_key(left, right), "bots": [left.label, right.label]}
            for left, right in pairings
        ],
        "wins": {},
        "side_wins": {},
        "winner_by_side": {},
        "pair_results": {},
        "turns": {},
        "fallback_reasons": {},
        "error_count": 0,
        "latest_error": None,
        "game_results": [],
    }
    _write_run_state(run_dir, state)

    try:
        game_number = 0
        for pair_index, (left, right) in enumerate(pairings, start=1):
            for game_in_pair in range(1, games_per_pair + 1):
                game_number += 1
                if game_in_pair % 2 == 1:
                    p1_spec, p2_spec = left, right
                else:
                    p1_spec, p2_spec = right, left
                state["current_game"] = {
                    "game": game_number,
                    "pair_game": game_in_pair,
                    "pair_index": pair_index,
                    "pair_key": _pair_key(left, right),
                    "started_at": _timestamp(),
                    "p1": p1_spec.label,
                    "p2": p2_spec.label,
                }
                _refresh_summary(state)
                _write_run_state(run_dir, state)
                result = _run_eval_game(
                    game=game_number,
                    pair_game=game_in_pair,
                    pair_index=pair_index,
                    left=left,
                    right=right,
                    p1_spec=p1_spec,
                    p2_spec=p2_spec,
                    project_root=project_root,
                    server_config=server_config,
                    base_agent=agents[0],
                )
                state["game_results"].append(result)
                state["games_completed"] = game_number
                state["current_game"] = None
                _refresh_summary(state)
                _write_run_state(run_dir, state)
                print(
                    (
                        f"progress {game_number}/{total_games}: "
                        f"pair={result['pair_key']} wins={state['wins']}, "
                        f"avg_turns={state['turns'].get('mean', 0)}"
                    ),
                    flush=True,
                )
        state["status"] = "complete"
        state["completed_at"] = _timestamp()
        _refresh_summary(state)
        _write_run_state(run_dir, state)
        return state
    except BaseException as error:
        state["status"] = "failed"
        state["completed_at"] = _timestamp()
        state["latest_error"] = f"{type(error).__name__}: {error}"
        _refresh_summary(state)
        _write_run_state(run_dir, state)
        raise


def list_eval_runs(runs_root: Path) -> list[Dict[str, Any]]:
    if not runs_root.exists():
        return []
    runs: list[Dict[str, Any]] = []
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        payload = _load_run_state(path)
        if payload is not None:
            runs.append(_run_summary(payload))
    return sorted(runs, key=lambda item: str(item.get("updated_at") or ""), reverse=True)


def load_eval_run(runs_root: Path, run_id: str) -> Optional[Dict[str, Any]]:
    return _load_run_state(runs_root / _safe_run_id(run_id))


def _resolve_bot_specs(bot_ids: list[str]) -> list[EvalBotSpec]:
    if len(bot_ids) < 2:
        raise ConfigError("At least two bots are required for a pool eval.")
    specs: list[EvalBotSpec] = []
    seen: set[str] = set()
    for bot_id in bot_ids:
        normalized = bot_id.strip()
        if not normalized:
            continue
        if normalized not in EVAL_BOT_SPECS:
            allowed = ", ".join(sorted(EVAL_BOT_SPECS))
            raise ConfigError(f"Unknown eval bot {normalized!r}. Available bots: {allowed}.")
        if normalized in seen:
            continue
        seen.add(normalized)
        specs.append(EVAL_BOT_SPECS[normalized])
    if len(specs) < 2:
        raise ConfigError("At least two distinct bots are required for a pool eval.")
    return specs


def _pairings(bot_specs: list[EvalBotSpec]) -> list[tuple[EvalBotSpec, EvalBotSpec]]:
    pairs: list[tuple[EvalBotSpec, EvalBotSpec]] = []
    for left_index, left in enumerate(bot_specs):
        for right in bot_specs[left_index + 1 :]:
            pairs.append((left, right))
    return pairs


def _pair_key(left: EvalBotSpec, right: EvalBotSpec) -> str:
    return " vs ".join(sorted((left.label, right.label)))


def _run_eval_game(
    *,
    game: int,
    pair_game: int,
    pair_index: int,
    left: EvalBotSpec,
    right: EvalBotSpec,
    p1_spec: EvalBotSpec,
    p2_spec: EvalBotSpec,
    project_root: Path,
    server_config,
    base_agent,
) -> Dict[str, Any]:
    specs = {"p1": p1_spec, "p2": p2_spec}

    agents = {slot: _eval_agent(base_agent, spec, slot, project_root) for slot, spec in specs.items()}
    labels = {slot: spec.label for slot, spec in specs.items()}
    battle_id = f"eval-gen1-pool-{game}"
    seed = [2026, 528, pair_index, pair_game]
    adapter = SimStreamAdapter(
        server_config=server_config,
        format_id=GEN1_RANDBAT_FORMAT,
        battle_id=battle_id,
        player_names=labels,
        seed=seed,
    )
    sessions = {
        slot: BattleSession(battle_id=battle_id, player_slot=slot, history_limit=agent.hook.history_turn_limit)
        for slot, agent in agents.items()
    }
    cursors = {slot: AgentContextCursor(last_turn_number=None, last_request_sequence=0) for slot in agents}
    rngs = {slot: random.Random(f"{seed}:{slot}:{specs[slot].bot_id}") for slot in agents}
    decisions: Counter[str] = Counter()
    fallback_reasons: defaultdict[str, Counter[str]] = defaultdict(Counter)
    errors: list[str] = []
    battle_log: list[str] = []
    started = time.monotonic()
    adapter.connect()
    try:
        for _ in range(10_000):
            event = adapter.next_event()
            for session in sessions.values():
                session.ingest(event)
            if event.event_type == "public_update":
                lines = event.payload.get("lines")
                if isinstance(lines, list):
                    battle_log.extend(line for line in lines if isinstance(line, str))
                continue
            if event.event_type == "request_received":
                slot = event.player_slot
                if slot not in agents:
                    continue
                session = sessions[slot]
                if not session.waiting_for_decision:
                    continue
                context = session.build_turn_context(agent=agents[slot], cursor=cursors[slot])
                choice, fallback_reason, error = _eval_choice(
                    spec=specs[slot],
                    context=context,
                    capture_payload=asdict(session.to_capture()),
                    rng=rngs[slot],
                    project_root=project_root,
                )
                bot_label = labels[slot]
                decisions[bot_label] += 1
                if fallback_reason:
                    fallback_reasons[bot_label][fallback_reason] += 1
                if error:
                    errors.append(f"{bot_label} {slot} turn {session.turn_number}: {error}")
                adapter.submit_decision(player_slot=slot, choice=choice, rqid=session.current_rqid)
                _submit_to_sessions(sessions, battle_id, slot, choice)
                cursors[slot] = session.advance_cursor()
                continue
            if event.event_type == "battle_finished":
                winner = _winner_from_finished_payload(event.payload, battle_log)
                return {
                    "game": game,
                    "pair_game": pair_game,
                    "pair_index": pair_index,
                    "pair_key": _pair_key(left, right),
                    "seed": seed,
                    "p1": labels["p1"],
                    "p2": labels["p2"],
                    "winner": winner,
                    "winner_side": "p1" if winner == labels["p1"] else ("p2" if winner == labels["p2"] else None),
                    "turns": sessions["p1"].turn_number or 0,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "decisions": dict(decisions),
                    "fallback_reasons": {bot: dict(counts) for bot, counts in fallback_reasons.items()},
                    "errors": errors[:10],
                    "battle_log": battle_log,
                }
        raise ConfigError("Eval game exceeded 10000 simulator events.")
    finally:
        adapter.close()


def _eval_choice(*, spec: EvalBotSpec, context, capture_payload: Dict[str, Any], rng: random.Random, project_root: Path):
    try:
        context_payload = asdict(context)
        if spec.bot_id == "max-damage-bot":
            plan = build_max_damage_bot_plan(
                context_payload,
                capture_payload=capture_payload,
                project_root=project_root,
            )
            return plan.decision, plan.fallback_reason, None
        if spec.hook_command == "custom-bot":
            plan = build_custom_bot_plan(
                context_payload,
                capture_payload=capture_payload,
                project_root=project_root,
                rng=rng,
                selection_strategy=spec.custom_selection_strategy or DEFAULT_CUSTOM_BOT_SELECTION_STRATEGY,
            )
            return plan.decision, plan.fallback_reason, None
        if spec.bot_id == "random-bot":
            decision = build_random_bot_decision(context_payload, rng=rng)
            return decision.decision, None, None
        raise ConfigError(f"Unsupported eval bot {spec.bot_id!r}.")
    except Exception as error:
        return choose_first_legal(context.request), "exception", f"{type(error).__name__}: {error}"


def _winner_from_finished_payload(payload: Dict[str, Any], battle_log: list[str]) -> str:
    winner = payload.get("winner") if isinstance(payload, dict) else None
    if isinstance(winner, str) and winner.strip():
        return winner.strip()
    if isinstance(payload, dict) and payload.get("tie"):
        return "tie"
    for line in reversed(battle_log):
        if line.startswith("|win|"):
            resolved = line[len("|win|") :].strip()
            if resolved:
                return resolved
        if line == "|tie":
            return "tie"
    return "unknown"


def _eval_agent(base_agent, spec: EvalBotSpec, slot: str, project_root: Path):
    python_command = str(project_root / ".venv" / "bin" / "python")
    if not Path(python_command).exists():
        python_command = "python3"
    args = ["-m", "pokerena", "agent", spec.hook_command]
    if spec.hook_command == "custom-bot" and spec.custom_selection_strategy:
        args.extend(["--selection-strategy", spec.custom_selection_strategy])
    return replace(
        base_agent,
        agent_id=f"gen1-{spec.bot_id}",
        provider=spec.provider,
        player_slot=slot,
        transport="sim-stream",
        launch=AgentLaunchConfig(
            command=python_command,
            args=args,
            cwd=project_root,
        ),
        callable=AgentCallableConfig(False, spec.label, [GEN1_RANDBAT_FORMAT], "accept-direct-challenges", None),
    )


def _submit_to_sessions(sessions: Dict[str, BattleSession], battle_id: str, player_slot: str, choice: str) -> None:
    event = SessionEvent(
        event_type="choice_submitted",
        battle_id=battle_id,
        player_slot=player_slot,
        payload={"choice": choice, "rqid": None},
    )
    for session in sessions.values():
        session.ingest(event)


def _refresh_summary(state: Dict[str, Any]) -> None:
    results = state.get("game_results") if isinstance(state.get("game_results"), list) else []
    wins = Counter(str(row.get("winner") or "unknown") for row in results if isinstance(row, dict))
    side_wins = Counter(str(row.get("winner_side") or "unknown") for row in results if isinstance(row, dict))
    winner_by_side = Counter(
        (str(row.get("winner") or "unknown"), str(row.get("winner_side") or "unknown"))
        for row in results
        if isinstance(row, dict)
    )
    turns = [int(row.get("turns") or 0) for row in results if isinstance(row, dict) and int(row.get("turns") or 0) > 0]
    combined_fallbacks: defaultdict[str, Counter[str]] = defaultdict(Counter)
    pair_results: dict[str, dict[str, Any]] = {}
    error_count = 0
    latest_error = state.get("latest_error")
    for row in results:
        if not isinstance(row, dict):
            continue
        pair_key = str(row.get("pair_key") or "")
        if pair_key:
            pair_state = pair_results.setdefault(
                pair_key,
                {
                    "pair_key": pair_key,
                    "bots": [row.get("p1"), row.get("p2")],
                    "games": 0,
                    "wins": {},
                    "win_rates": {},
                    "turns": [],
                    "mean_turns": None,
                },
            )
            pair_state["games"] += 1
            winner = str(row.get("winner") or "unknown")
            pair_state["wins"][winner] = pair_state["wins"].get(winner, 0) + 1
            if int(row.get("turns") or 0) > 0:
                pair_state["turns"].append(int(row.get("turns") or 0))
        for error in row.get("errors") or []:
            latest_error = error
            error_count += 1
        fallback_reasons = row.get("fallback_reasons")
        if isinstance(fallback_reasons, dict):
            for bot, reasons in fallback_reasons.items():
                if isinstance(reasons, dict):
                    combined_fallbacks[str(bot)].update({str(key): int(value) for key, value in reasons.items()})
    state["wins"] = dict(wins)
    state["side_wins"] = dict(side_wins)
    state["winner_by_side"] = {f"{winner}:{side}": count for (winner, side), count in winner_by_side.items()}
    for pair_state in pair_results.values():
        games = int(pair_state.get("games") or 0)
        pair_state["win_rates"] = {
            bot: round((wins_count / games) * 100.0, 1)
            for bot, wins_count in pair_state.get("wins", {}).items()
            if games > 0
        }
        pair_turns = pair_state.pop("turns", [])
        pair_state["mean_turns"] = round(statistics.mean(pair_turns), 2) if pair_turns else None
    state["pair_results"] = dict(sorted(pair_results.items()))
    state["turns"] = {
        "min": min(turns) if turns else None,
        "max": max(turns) if turns else None,
        "mean": round(statistics.mean(turns), 2) if turns else None,
        "median": statistics.median(turns) if turns else None,
    }
    state["fallback_reasons"] = {bot: dict(counts) for bot, counts in combined_fallbacks.items()}
    state["error_count"] = error_count
    state["latest_error"] = latest_error
    state["updated_at"] = _timestamp()


def _run_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "run_id",
        "kind",
        "status",
        "format",
        "games_total",
        "games_completed",
        "current_game",
        "started_at",
        "updated_at",
        "completed_at",
        "bots",
        "wins",
        "pair_results",
        "turns",
        "error_count",
        "latest_error",
    ]
    return {key: payload.get(key) for key in keys}


def _write_run_state(run_dir: Path, state: Dict[str, Any]) -> None:
    path = run_dir / "run.json"
    tmp_path = run_dir / "run.json.tmp"
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _load_run_state(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "run.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _safe_run_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-")
    if not safe:
        raise ConfigError("run_id must contain at least one safe character.")
    return safe


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _build_eval_handler(runs_root: Path):
    class EvalViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in {"/", "/index.html"}:
                self._send_html(_INDEX_HTML)
                return
            if path == "/healthz":
                self._send_json({"ok": True})
                return
            if path == "/api/runs":
                self._send_json({"runs": list_eval_runs(runs_root)})
                return
            if path.startswith("/api/runs/"):
                run_id = unquote(path[len("/api/runs/") :])
                payload = load_eval_run(runs_root, run_id)
                if payload is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "Eval run not found.")
                    return
                self._send_json(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found.")

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _send_html(self, content: str) -> None:
            body = content.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return EvalViewerHandler


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pokerena Eval Tracker</title>
  <style>
    :root {
      --bg: #101411;
      --panel: #f3ead8;
      --panel-2: #fff7e8;
      --ink: #171711;
      --muted: #756d5e;
      --line: #d5c5a7;
      --green: #2f7a4f;
      --red: #9d3d30;
      --gold: #c2872e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 15% 5%, rgba(231, 184, 91, 0.24), transparent 30rem),
        linear-gradient(135deg, #101411, #273529 55%, #171915);
    }
    button { font: inherit; color: inherit; cursor: pointer; }
    .shell { display: grid; grid-template-columns: 340px 1fr; min-height: 100vh; }
    .sidebar {
      padding: 28px 18px;
      background: rgba(16, 20, 17, 0.88);
      color: #f5ecd8;
      border-right: 1px solid rgba(255, 255, 255, 0.14);
      overflow: auto;
    }
    h1 { margin: 0 0 8px; font-size: 30px; }
    .sidebar p { margin: 0 0 20px; color: #c4b9a4; line-height: 1.45; }
    .run-list { display: grid; gap: 10px; }
    .run-card {
      width: 100%;
      text-align: left;
      border: 1px solid rgba(255,255,255,0.16);
      border-radius: 18px;
      padding: 14px;
      background: rgba(255,255,255,0.07);
      color: inherit;
    }
    .run-card.active { border-color: #e5b55f; background: rgba(229, 181, 95, 0.17); }
    .run-card strong { display: block; margin-bottom: 6px; }
    .muted { color: var(--muted); }
    .sidebar .muted { color: #c4b9a4; }
    .main { padding: 32px; overflow: auto; }
    .hero {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 24px;
      box-shadow: 0 22px 70px rgba(0,0,0,0.22);
      margin-bottom: 18px;
    }
    .hero-top { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; }
    h2 { margin: 0 0 6px; font-size: 34px; }
    .status { display: inline-flex; align-items: center; gap: 8px; padding: 7px 12px; border-radius: 999px; background: #e7dcc4; }
    .status.running { background: #d8ead7; color: #1e5d37; }
    .status.complete { background: #d9e4f2; color: #284c7b; }
    .status.failed { background: #f0d0cb; color: #8a3026; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin-top: 18px; }
    .metric {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
    }
    .metric span { display: block; color: var(--muted); font-size: 13px; margin-bottom: 8px; }
    .metric strong { font-size: 28px; }
    .bar { height: 14px; border-radius: 999px; overflow: hidden; background: #dbcdb1; margin-top: 16px; }
    .bar > div { height: 100%; background: linear-gradient(90deg, var(--green), #93b65d); width: 0%; transition: width 240ms ease; }
    .panels { display: grid; grid-template-columns: minmax(0, 1fr) minmax(320px, 0.45fr); gap: 18px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 18px;
      box-shadow: 0 18px 60px rgba(0,0,0,0.16);
    }
    .panel h3 { margin: 0 0 12px; font-size: 22px; }
    .panel h3:not(:first-child) { margin-top: 24px; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { border-bottom: 1px solid #dcccae; padding: 9px 6px; text-align: left; }
    th { color: var(--muted); font-weight: 600; }
    tbody tr { cursor: pointer; }
    tbody tr:hover { background: rgba(194, 135, 46, 0.12); }
    tbody tr.selected { background: rgba(47, 122, 79, 0.14); }
    .winner { font-weight: 700; color: var(--green); }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: #fff7e8;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
      max-height: 360px;
      overflow: auto;
    }
    @media (max-width: 920px) {
      .shell { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid rgba(255,255,255,0.14); max-height: 42vh; }
      .panels { grid-template-columns: 1fr; }
      .hero-top { flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <h1>Eval Tracker</h1>
      <p>Live headless battle runs from Pokerena simulate-battle.</p>
      <div id="run-list" class="run-list"></div>
    </aside>
    <main class="main">
      <section class="hero">
        <div class="hero-top">
          <div>
            <h2 id="title">No run selected</h2>
            <div id="subtitle" class="muted">Start an eval run and it will appear here.</div>
          </div>
          <div id="status" class="status">idle</div>
        </div>
        <div class="bar"><div id="progress-bar"></div></div>
        <div class="grid">
          <div class="metric"><span>Progress</span><strong id="progress">0 / 0</strong></div>
          <div class="metric"><span>Leader</span><strong id="leader">-</strong></div>
          <div class="metric"><span>Bots</span><strong id="bot-count">0</strong></div>
          <div class="metric"><span>Average turns</span><strong id="avg-turns">-</strong></div>
        </div>
      </section>
      <div class="panels">
        <section class="panel">
          <h3>Pair Win Rates</h3>
          <table>
            <thead><tr><th>Pair</th><th>Games</th><th>Wins</th><th>Win rates</th></tr></thead>
            <tbody id="pair-results"></tbody>
          </table>
          <h3>Games</h3>
          <table>
            <thead><tr><th>Game</th><th>Pair</th><th>Winner</th><th>Turns</th><th>P1</th><th>P2</th></tr></thead>
            <tbody id="games"></tbody>
          </table>
        </section>
        <section class="panel">
          <h3 id="details-title">Run Details</h3>
          <pre id="details">{}</pre>
        </section>
      </div>
    </main>
  </div>
  <script>
    let selectedRunId = null;
    let selectedGame = null;

    async function fetchJson(url) {
      const response = await fetch(url, {cache: "no-store"});
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return response.json();
    }

    function fmt(value) {
      return value === null || value === undefined || value === "" ? "-" : String(value);
    }

    function setText(id, value) {
      document.getElementById(id).textContent = fmt(value);
    }

    async function refresh() {
      try {
        const list = await fetchJson("/api/runs");
        renderRunList(list.runs || []);
        if (!selectedRunId && list.runs && list.runs.length) selectedRunId = list.runs[0].run_id;
        if (selectedRunId) {
          const run = await fetchJson(`/api/runs/${encodeURIComponent(selectedRunId)}`);
          renderRun(run);
        }
      } catch (error) {
        setText("subtitle", `Disconnected: ${error.message}`);
      }
    }

    function renderRunList(runs) {
      const list = document.getElementById("run-list");
      const existing = new Map([...list.querySelectorAll(".run-card")].map(node => [node.dataset.runId, node]));
      for (const run of runs) {
        let card = existing.get(run.run_id);
        if (!card) {
          card = document.createElement("button");
          card.type = "button";
          card.className = "run-card";
          card.dataset.runId = run.run_id;
          card.addEventListener("click", () => {
            selectedRunId = run.run_id;
            selectedGame = null;
            refresh();
          });
          list.appendChild(card);
        }
        card.classList.toggle("active", run.run_id === selectedRunId);
        card.innerHTML = `<strong>${escapeHtml(run.run_id)}</strong><div class="muted">${escapeHtml(run.status)} · ${run.games_completed || 0}/${run.games_total || 0}<br>${escapeHtml(run.updated_at || "")}</div>`;
        existing.delete(run.run_id);
      }
      for (const node of existing.values()) node.remove();
    }

    function renderRun(run) {
      const total = run.games_total || 0;
      const done = run.games_completed || 0;
      const pct = total ? Math.round(done / total * 100) : 0;
      setText("title", run.run_id);
      setText("subtitle", `${run.kind || "eval"} · ${run.format || ""} · updated ${run.updated_at || "-"}`);
      const status = document.getElementById("status");
      status.textContent = run.status || "unknown";
      status.className = `status ${run.status || ""}`;
      document.getElementById("progress-bar").style.width = `${pct}%`;
      setText("progress", `${done} / ${total}`);
      const wins = run.wins || {};
      const sortedWins = Object.entries(wins).sort((left, right) => (right[1] || 0) - (left[1] || 0));
      setText("leader", sortedWins.length ? `${sortedWins[0][0]} (${sortedWins[0][1]})` : "-");
      setText("bot-count", (run.bots || []).length);
      setText("avg-turns", run.turns && run.turns.mean);
      renderPairResults(run.pair_results || {});
      const games = document.getElementById("games");
      games.innerHTML = "";
      for (const game of (run.game_results || []).slice().reverse()) {
        const row = document.createElement("tr");
        row.className = game.game === selectedGame ? "selected" : "";
        row.addEventListener("click", () => {
          selectedGame = game.game;
          renderRun(run);
        });
        row.innerHTML = `<td>${game.game}</td><td>${escapeHtml(game.pair_key || "-")}</td><td class="winner">${escapeHtml(game.winner || "-")}</td><td>${game.turns || "-"}</td><td>${escapeHtml(game.p1 || "-")}</td><td>${escapeHtml(game.p2 || "-")}</td>`;
        games.appendChild(row);
      }
      const selected = (run.game_results || []).find(game => game.game === selectedGame);
      if (selected && Array.isArray(selected.battle_log)) {
        document.getElementById("details-title").textContent = `Battle Log · Game ${selected.game}`;
        document.getElementById("details").textContent = selected.battle_log.join("\\n") || "No public battle log recorded.";
      } else if (selected) {
        document.getElementById("details-title").textContent = `Battle Log · Game ${selected.game}`;
        document.getElementById("details").textContent = "This run was recorded before battle logs were captured. Rerun the eval to populate logs.";
      } else {
        document.getElementById("details-title").textContent = "Run Details";
        const summary = {
          wins: run.wins,
          side_wins: run.side_wins,
          winner_by_side: run.winner_by_side,
          pair_results: run.pair_results,
          turns: run.turns,
          fallback_reasons: run.fallback_reasons,
          error_count: run.error_count,
          latest_error: run.latest_error,
          current_game: run.current_game,
        };
        document.getElementById("details").textContent = JSON.stringify(summary, null, 2);
      }
    }

    function renderPairResults(pairResults) {
      const tbody = document.getElementById("pair-results");
      tbody.innerHTML = "";
      const rows = Object.values(pairResults).sort((left, right) => String(left.pair_key || "").localeCompare(String(right.pair_key || "")));
      for (const pair of rows) {
        const wins = Object.entries(pair.wins || {}).map(([bot, count]) => `${bot}: ${count}`).join(", ");
        const rates = Object.entries(pair.win_rates || {}).map(([bot, rate]) => `${bot}: ${rate}%`).join(", ");
        const row = document.createElement("tr");
        row.innerHTML = `<td>${escapeHtml(pair.pair_key || "-")}</td><td>${pair.games || 0}</td><td>${escapeHtml(wins || "-")}</td><td>${escapeHtml(rates || "-")}</td>`;
        tbody.appendChild(row);
      }
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[char]));
    }

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""
