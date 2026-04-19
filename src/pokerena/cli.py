from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import List, Optional, Sequence

from .agent import (
    AgentContextCursor,
    BattleSession,
    DecisionPolicy,
    SessionEvent,
    SimStreamAdapter,
    build_session_from_capture,
    choose_first_legal,
    default_capture_path,
    default_cursor_path,
    find_agent,
    invoke_agent,
    load_capture,
    load_cursor,
    save_capture,
    save_cursor,
)
from .config import ConfigError, load_agents_config, load_server_config
from .showdown import build_server_command, node_version, prepare_runtime


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.command == "server":
            if args.server_command == "doctor":
                return run_doctor(args)
            if args.server_command == "render-config":
                return run_render_config(args)
            if args.server_command == "up":
                return run_up(args)
        if args.command == "agent":
            if args.agent_command == "context":
                return run_agent_context(args)
            if args.agent_command == "decide":
                return run_agent_decide(args)
            if args.agent_command == "sim-battle":
                return run_agent_sim_battle(args)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    parser.error("A command is required.")
    return 2


def main_entry() -> None:
    raise SystemExit(main())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pokerena")
    subparsers = parser.add_subparsers(dest="command")

    server_parser = subparsers.add_parser("server", help="Manage the local Pokerena server scaffold.")
    server_subparsers = server_parser.add_subparsers(dest="server_command")

    doctor_parser = server_subparsers.add_parser("doctor", help="Check the local environment.")
    _add_common_config_arguments(doctor_parser)

    render_parser = server_subparsers.add_parser(
        "render-config",
        help="Render the generated Pokemon Showdown config into .runtime/showdown.",
    )
    _add_common_config_arguments(render_parser, include_agents=False)

    up_parser = server_subparsers.add_parser("up", help="Start the local Pokemon Showdown server.")
    _add_common_config_arguments(up_parser, include_agents=False)
    up_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render runtime files and print the launch command without starting the process.",
    )

    agent_parser = subparsers.add_parser("agent", help="Manage the battle-agent runtime.")
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command")

    context_parser = agent_subparsers.add_parser(
        "context",
        help="Render the stable turn-context payload for an agent from a recorded battle capture.",
    )
    _add_agent_replay_arguments(context_parser)

    decide_parser = agent_subparsers.add_parser(
        "decide",
        help="Build a turn context from a recorded capture and invoke the configured agent hook.",
    )
    _add_agent_replay_arguments(decide_parser)
    decide_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write context and prompt files without invoking the configured agent command.",
    )
    decide_parser.add_argument(
        "--decision-timeout",
        type=int,
        default=45,
        help="How long to wait for the agent subprocess before treating it as timed out.",
    )

    sim_parser = agent_subparsers.add_parser(
        "sim-battle",
        help="Run a live local simulator battle with a configured agent on one side.",
    )
    _add_common_config_arguments(sim_parser)
    sim_parser.add_argument("--agent-id", required=True, help="Agent identifier from the agents config.")
    sim_parser.add_argument(
        "--format",
        default=None,
        help="Simulator format to run. Defaults to the first format_allowlist entry for the agent.",
    )
    sim_parser.add_argument(
        "--battle-id",
        default=None,
        help="Optional explicit battle identifier for runtime artifacts and capture output.",
    )
    sim_parser.add_argument(
        "--capture",
        default=None,
        help="Optional path to save the normalized battle capture JSON.",
    )
    sim_parser.add_argument(
        "--cursor",
        default=None,
        help="Optional path to save the agent cursor state.",
    )
    sim_parser.add_argument(
        "--history-limit",
        type=int,
        default=60,
        help="How many recent public protocol lines to keep in each turn context.",
    )
    sim_parser.add_argument(
        "--decision-timeout",
        type=int,
        default=45,
        help="How long to wait for the agent subprocess before falling back.",
    )
    sim_parser.add_argument(
        "--max-invalid-retries",
        type=int,
        default=3,
        help="How many invalid agent decisions to tolerate before falling back to the built-in policy.",
    )
    sim_parser.add_argument(
        "--seed",
        default=None,
        help="Optional simulator seed as four comma-separated integers, for example 1,2,3,4.",
    )
    sim_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stop after the first agent turn context is written.",
    )

    return parser


def _add_common_config_arguments(parser: argparse.ArgumentParser, include_agents: bool = True) -> None:
    parser.add_argument(
        "--config",
        default="config/server.local.yaml",
        help="Path to the Pokerena server config file.",
    )
    if include_agents:
        parser.add_argument(
            "--agents-config",
            default="config/agents.yaml",
            help="Path to the Pokerena agents config file.",
        )


def _add_agent_replay_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent-id", required=True, help="Agent identifier from the agents config.")
    parser.add_argument(
        "--agents-config",
        default="config/agents.yaml",
        help="Path to the Pokerena agents config file.",
    )
    parser.add_argument(
        "--capture",
        required=True,
        help="Path to a Pokerena battle capture JSON file.",
    )
    parser.add_argument(
        "--cursor",
        default=None,
        help="Optional path to a persisted agent cursor. Defaults to an empty cursor if omitted.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=60,
        help="How many recent public protocol lines to keep in the reconstructed context.",
    )


def run_doctor(args: argparse.Namespace) -> int:
    results = collect_doctor_checks(
        config_path=args.config,
        agents_config_path=args.agents_config,
        project_root=Path.cwd(),
    )
    for result in results:
        status = "PASS" if result.ok else ("WARN" if not result.required else "FAIL")
        print(f"[{status}] {result.name}: {result.detail}")

    failures = [result for result in results if result.required and not result.ok]
    return 1 if failures else 0


def run_render_config(args: argparse.Namespace) -> int:
    server_config = load_server_config(config_path=args.config)
    artifacts = prepare_runtime(server_config)
    print(f"Rendered runtime config: {artifacts.runtime_config_path}")
    print(f"Linked Showdown config: {artifacts.submodule_config_path}")
    print(f"Wrote runtime metadata: {artifacts.runtime_metadata_path}")
    return 0


def run_up(args: argparse.Namespace) -> int:
    server_config = load_server_config(config_path=args.config)
    artifacts = prepare_runtime(server_config)
    command = build_server_command(server_config)

    print(f"Runtime config: {artifacts.runtime_config_path}")
    print(f"Showdown config link: {artifacts.submodule_config_path}")
    print("Command:", " ".join(command))

    if args.dry_run:
        return 0

    if shutil.which("node") is None:
        print("error: node is required on PATH. Run scripts/bootstrap-showdown.sh first.", file=sys.stderr)
        return 1

    version = node_version()
    if not _is_supported_node_version(version):
        print(f"error: Node.js 22+ is required; found {version or 'no node installation'}.", file=sys.stderr)
        return 1

    if shutil.which("npm") is None:
        print("error: npm is required on PATH. Run scripts/bootstrap-showdown.sh first.", file=sys.stderr)
        return 1

    launcher_path = server_config.showdown_path / "pokemon-showdown"
    if not launcher_path.exists():
        print(
            f"error: Showdown launcher not found at {launcher_path}. Initialize the submodule first.",
            file=sys.stderr,
        )
        return 1

    node_modules = server_config.showdown_path / "node_modules"
    if not node_modules.exists():
        print(
            f"error: {node_modules} is missing. Run scripts/bootstrap-showdown.sh first.",
            file=sys.stderr,
        )
        return 1

    completed = subprocess.run(command, cwd=server_config.showdown_path)
    return completed.returncode


def run_agent_context(args: argparse.Namespace) -> int:
    context, _, _, _ = _load_replay_context(args)
    print(json.dumps(asdict(context), indent=2))
    return 0


def run_agent_decide(args: argparse.Namespace) -> int:
    context, agent, session, cursor_path = _load_replay_context(args)
    runtime_root = Path.cwd() / ".runtime"
    capture_path = Path(args.capture).resolve()
    decision, artifacts = invoke_agent(
        agent=agent,
        context=context,
        runtime_root=runtime_root,
        capture_path=capture_path,
        dry_run=args.dry_run,
        timeout_seconds=args.decision_timeout,
    )

    print(f"Context: {artifacts.context_path}")
    print(f"Prompt: {artifacts.prompt_path}")
    print(f"Capture: {capture_path}")
    print(f"Turn started: {context.signals['turn_started']}")
    print(f"Request updated: {context.signals['request_updated']}")

    if args.dry_run:
        print("Agent invocation skipped (--dry-run).")
        return 0

    if decision is None:
        raise ConfigError("Agent invocation did not return a decision.")

    save_cursor(cursor_path, session.advance_cursor())
    print(f"Response: {artifacts.response_path}")
    print(json.dumps(asdict(decision), indent=2))
    return 0


def run_agent_sim_battle(args: argparse.Namespace) -> int:
    server_config = load_server_config(config_path=args.config, project_root=Path.cwd())
    agents = load_agents_config(config_path=args.agents_config, project_root=Path.cwd())
    agent = find_agent(agents, args.agent_id)
    if not agent.enabled:
        raise ConfigError(f"Agent {agent.agent_id!r} is disabled in the agents config.")
    if agent.transport != "sim-stream":
        raise ConfigError(
            f"Agent {agent.agent_id!r} uses transport {agent.transport!r}. The live runtime currently supports sim-stream only."
        )

    format_id = _resolve_format_id(agent, args.format)
    battle_id = args.battle_id or f"sim-{format_id}-{int(time.time())}"
    runtime_root = Path.cwd() / ".runtime"
    capture_path = (
        Path(args.capture).resolve()
        if args.capture
        else default_capture_path(runtime_root=runtime_root, agent=agent, battle_id=battle_id)
    )
    cursor_path = (
        Path(args.cursor).resolve()
        if args.cursor
        else default_cursor_path(runtime_root=runtime_root, agent=agent, battle_id=battle_id)
    )
    cursor = load_cursor(cursor_path)
    policy = DecisionPolicy(
        max_invalid_retries=args.max_invalid_retries,
        decision_timeout_seconds=args.decision_timeout,
        history_limit=args.history_limit,
    )
    session = BattleSession(
        battle_id=battle_id,
        player_slot=agent.player_slot,
        history_limit=policy.history_limit,
    )
    adapter = SimStreamAdapter(
        server_config=server_config,
        format_id=format_id,
        battle_id=battle_id,
        player_names={
            "p1": agent.agent_id if agent.player_slot == "p1" else "first-legal-bot",
            "p2": agent.agent_id if agent.player_slot == "p2" else "first-legal-bot",
        },
        seed=_parse_seed(args.seed),
    )

    try:
        adapter.connect()
        while True:
            event = adapter.next_event()
            session.ingest(event)

            if event.event_type == "request_received":
                if event.player_slot == agent.player_slot and session.waiting_for_decision:
                    context = session.build_turn_context(agent=agent, cursor=cursor)
                    if args.dry_run:
                        _, artifacts = invoke_agent(
                            agent=agent,
                            context=context,
                            runtime_root=runtime_root,
                            capture_path=capture_path,
                            dry_run=True,
                            timeout_seconds=policy.decision_timeout_seconds,
                        )
                        save_capture(capture_path, session.to_capture())
                        print(f"Context: {artifacts.context_path}")
                        print(f"Prompt: {artifacts.prompt_path}")
                        print(f"Capture: {capture_path}")
                        print("Agent invocation skipped (--dry-run).")
                        return 0

                    decision_text = _decide_or_fallback(
                        agent=agent,
                        session=session,
                        context=context,
                        runtime_root=runtime_root,
                        capture_path=capture_path,
                        policy=policy,
                    )
                    _submit_choice(
                        adapter=adapter,
                        session=session,
                        player_slot=agent.player_slot,
                        choice=decision_text,
                        rqid=session.current_rqid,
                    )
                    cursor = session.advance_cursor()
                    save_cursor(cursor_path, cursor)
                    continue

                if event.player_slot == _opponent_slot(agent.player_slot):
                    choice = choose_first_legal(event.payload)
                    _submit_choice(
                        adapter=adapter,
                        session=session,
                        player_slot=event.player_slot,
                        choice=choice,
                        rqid=None,
                    )
                    continue

            if event.event_type == "battle_finished":
                save_capture(capture_path, session.to_capture())
                winner = event.payload.get("winner")
                if winner:
                    print(f"Battle finished: winner={winner}")
                print(f"Capture: {capture_path}")
                return 0
    finally:
        save_capture(capture_path, session.to_capture())
        adapter.close()


def _load_replay_context(args: argparse.Namespace):
    agents = load_agents_config(config_path=args.agents_config, project_root=Path.cwd())
    agent = find_agent(agents, args.agent_id)
    capture_path = Path(args.capture).resolve()
    capture = load_capture(capture_path)
    session = build_session_from_capture(
        capture=capture,
        player_slot=agent.player_slot,
        history_limit=args.history_limit,
    )
    cursor_path = (
        Path(args.cursor).resolve()
        if args.cursor
        else default_cursor_path(Path.cwd() / ".runtime", agent=agent, battle_id=capture.battle_id)
    )
    cursor = load_cursor(cursor_path)
    context = session.build_turn_context(agent=agent, cursor=cursor)
    return context, agent, session, cursor_path


def _decide_or_fallback(
    *,
    agent,
    session: BattleSession,
    context,
    runtime_root: Path,
    capture_path: Path,
    policy: DecisionPolicy,
) -> str:
    if session.current_invalid_attempts() >= policy.max_invalid_retries:
        return choose_first_legal(session.current_request)

    try:
        decision, _ = invoke_agent(
            agent=agent,
            context=context,
            runtime_root=runtime_root,
            capture_path=capture_path,
            dry_run=False,
            timeout_seconds=policy.decision_timeout_seconds,
        )
    except ConfigError:
        return choose_first_legal(session.current_request)

    if decision is None:
        return choose_first_legal(session.current_request)
    return decision.decision


def _submit_choice(
    *,
    adapter: SimStreamAdapter,
    session: BattleSession,
    player_slot: str,
    choice: str,
    rqid: Optional[str],
) -> None:
    adapter.submit_decision(player_slot=player_slot, choice=choice, rqid=rqid)
    session.ingest(
        SessionEvent(
            event_type="choice_submitted",
            battle_id=session.battle_id,
            player_slot=player_slot,
            payload={"choice": choice, "rqid": rqid},
        )
    )


def _resolve_format_id(agent, explicit_format: Optional[str]) -> str:
    if explicit_format:
        if agent.format_allowlist and explicit_format not in agent.format_allowlist:
            allowed = ", ".join(agent.format_allowlist)
            raise ConfigError(
                f"Format {explicit_format!r} is not allowed for agent {agent.agent_id!r}. Allowed formats: {allowed}."
            )
        return explicit_format
    if agent.format_allowlist:
        return agent.format_allowlist[0]
    raise ConfigError(
        f"Agent {agent.agent_id!r} has no format_allowlist and no --format override was provided."
    )


def _parse_seed(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 4:
        raise ConfigError("--seed must contain exactly four comma-separated integers.")
    try:
        return [int(item) for item in parts]
    except ValueError as error:
        raise ConfigError("--seed must contain only integers.") from error


def _opponent_slot(player_slot: str) -> str:
    return "p2" if player_slot == "p1" else "p1"


def collect_doctor_checks(
    config_path: str,
    agents_config_path: str,
    project_root: Path,
) -> List[CheckResult]:
    results: List[CheckResult] = []

    py_ok = sys.version_info[:2] == (3, 14)
    results.append(
        CheckResult(
            name="python",
            ok=py_ok,
            detail=f"running {sys.version.split()[0]} (expected 3.14.x)",
        )
    )

    try:
        server_config = load_server_config(config_path=config_path, project_root=project_root)
        results.append(CheckResult("server-config", True, f"loaded {server_config.config_path}"))
    except ConfigError as error:
        server_config = None
        results.append(CheckResult("server-config", False, str(error)))

    try:
        agents = load_agents_config(config_path=agents_config_path, project_root=project_root)
        results.append(CheckResult("agents-config", True, f"loaded {len(agents)} agent definition(s)"))
    except ConfigError as error:
        results.append(CheckResult("agents-config", False, str(error)))

    node_binary = shutil.which("node")
    version = node_version() if node_binary else None
    results.append(
        CheckResult(
            name="node",
            ok=_is_supported_node_version(version),
            detail=version if version else "node is not installed",
        )
    )
    npm_binary = shutil.which("npm")
    results.append(
        CheckResult(
            name="npm",
            ok=npm_binary is not None,
            detail=npm_binary or "npm is not installed",
        )
    )

    docker_binary = shutil.which("docker")
    results.append(
        CheckResult(
            name="docker",
            ok=docker_binary is not None,
            detail=docker_binary or "docker is not installed",
            required=False,
        )
    )

    if server_config is not None:
        launcher_path = server_config.showdown_path / "pokemon-showdown"
        node_modules = server_config.showdown_path / "node_modules"
        results.append(
            CheckResult(
                name="showdown-launcher",
                ok=launcher_path.exists(),
                detail=str(launcher_path) if launcher_path.exists() else f"missing {launcher_path}",
            )
        )
        results.append(
            CheckResult(
                name="showdown-deps",
                ok=node_modules.exists(),
                detail=str(node_modules) if node_modules.exists() else f"missing {node_modules}",
            )
        )

    return results


def _is_supported_node_version(version: Optional[str]) -> bool:
    if not version:
        return False
    normalized = version.removeprefix("v")
    major = normalized.split(".", 1)[0]
    return major.isdigit() and int(major) >= 22
