from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import IO, List, Optional, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from .agent import (
    AgentCancelledError,
    AgentTimeoutError,
    AgentContextCursor,
    AgentDecision,
    BattleSession,
    DecisionPolicy,
    SessionEvent,
    ShowdownClientAdapter,
    SimStreamAdapter,
    build_session_from_capture,
    choose_first_legal,
    choose_random_legal,
    default_capture_path,
    default_cursor_path,
    find_agent,
    invoke_agent,
    load_capture,
    load_cursor,
    render_turn_prompt,
    save_capture,
    save_cursor,
)
from .calc import (
    CALC_BATCH_REQUEST_SCHEMA_VERSION,
    CALC_DEPENDENCY_PATH,
    CALC_SCRIPT_PATH,
    DEFAULT_CALC_TIMEOUT_SECONDS,
    detect_project_root,
    read_damage_calc_batch_input,
    read_damage_calc_input,
    run_damage_calc_batch,
    run_damage_calc,
    sample_damage_calc_batch_payload,
    sample_damage_calc_payload,
)
from .config import ConfigError, load_agents_config, load_server_config
from .showdown import build_server_command, node_version, prepare_runtime
from .transcript import (
    TranscriptEntry,
    append_transcript_trace_event,
    clear_battle_stop_request,
    load_battle_stop_request,
    mark_battle_stop_handled,
    default_transcript_path,
    load_transcript_entry,
    record_transcript_entry,
    update_transcript_entry,
    update_transcript_entry_state,
    update_transcript_metadata,
    upsert_battle_summary_entry,
)
from .transcript_viewer import serve_transcript_viewer, transcript_viewer_url


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


class ManualBattleStopRequested(ConfigError):
    """Raised when a live battle is manually stopped from the viewer."""


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
            if args.server_command == "transcript-viewer":
                return run_transcript_viewer(args)
        if args.command == "calc":
            if args.calc_command == "damage":
                return run_calc_damage(args)
            if args.calc_command == "damage-batch":
                return run_calc_damage_batch(args)
        if args.command == "agent":
            if args.agent_command == "context":
                return run_agent_context(args)
            if args.agent_command == "decide":
                return run_agent_decide(args)
            if args.agent_command == "sim-battle":
                return run_agent_sim_battle(args)
            if args.agent_command == "showdown-client":
                return run_agent_showdown_client(args)
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
    _add_common_config_arguments(up_parser)
    up_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render runtime files and print the launch command without starting the process.",
    )
    viewer_parser = server_subparsers.add_parser(
        "transcript-viewer",
        help="Serve the local agent transcript viewer.",
    )
    _add_common_config_arguments(viewer_parser, include_agents=False)

    calc_parser = subparsers.add_parser("calc", help="Run local agent support tools.")
    calc_subparsers = calc_parser.add_subparsers(dest="calc_command")

    damage_parser = calc_subparsers.add_parser(
        "damage",
        help="Calculate damage using the local @smogon/calc wrapper.",
    )
    damage_input = damage_parser.add_mutually_exclusive_group(required=True)
    damage_input.add_argument(
        "--input",
        help="Path to a JSON file describing one damage calculation request.",
    )
    damage_input.add_argument(
        "--stdin",
        action="store_true",
        help="Read the damage calculation request JSON from stdin.",
    )
    damage_parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_CALC_TIMEOUT_SECONDS,
        help="How long to wait for the local Node damage calc process before failing.",
    )

    batch_parser = calc_subparsers.add_parser(
        "damage-batch",
        help="Calculate multiple damage rolls using the local persistent calc worker.",
    )
    batch_input = batch_parser.add_mutually_exclusive_group(required=True)
    batch_input.add_argument(
        "--input",
        help="Path to a JSON file describing a batch of damage calculation requests.",
    )
    batch_input.add_argument(
        "--stdin",
        action="store_true",
        help="Read the damage batch request JSON from stdin.",
    )
    batch_parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_CALC_TIMEOUT_SECONDS,
        help="How long to wait for the local Node damage calc worker before failing.",
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
        default=None,
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
        help="How many recent public battle turns to keep in memory while the battle runs.",
    )
    sim_parser.add_argument(
        "--decision-timeout",
        type=int,
        default=None,
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

    showdown_parser = agent_subparsers.add_parser(
        "showdown-client",
        help="Run one callable local agent against browser challenges on the local Showdown server.",
    )
    _add_common_config_arguments(showdown_parser)
    showdown_parser.add_argument("--agent-id", required=True, help="Agent identifier from the agents config.")
    showdown_parser.add_argument(
        "--history-limit",
        type=int,
        default=60,
        help="How many recent public protocol lines to keep in each turn context.",
    )
    showdown_parser.add_argument(
        "--decision-timeout",
        type=int,
        default=None,
        help="How long to wait for the agent subprocess before falling back.",
    )
    showdown_parser.add_argument(
        "--max-invalid-retries",
        type=int,
        default=3,
        help="How many invalid agent decisions to tolerate before falling back to the built-in policy.",
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
        help="How many recent public battle turns to keep in the reconstructed session state.",
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


def run_transcript_viewer(args: argparse.Namespace) -> int:
    server_config = load_server_config(config_path=args.config, project_root=Path.cwd())
    if not server_config.transcript_viewer.enabled:
        raise ConfigError("The transcript viewer is disabled in the loaded server config.")
    print(f"Transcript viewer: {transcript_viewer_url(server_config)}")
    serve_transcript_viewer(server_config)
    return 0


def run_up(args: argparse.Namespace) -> int:
    project_root = Path.cwd()
    server_config = load_server_config(config_path=args.config, project_root=project_root)
    agents = load_agents_config(config_path=args.agents_config, project_root=project_root)
    callable_agents = [agent for agent in agents if agent.enabled and agent.callable.enabled]
    artifacts = prepare_runtime(server_config)
    command = build_server_command(server_config)

    print(f"Runtime config: {artifacts.runtime_config_path}")
    print(f"Showdown config link: {artifacts.submodule_config_path}")
    print("Command:", " ".join(command))
    if server_config.transcript_viewer.enabled:
        print(f"Transcript viewer: {transcript_viewer_url(server_config)}")
    if callable_agents:
        print(
            "Callable agents:",
            ", ".join(f"{agent.agent_id} as {agent.callable.username}" for agent in callable_agents),
        )

    if args.dry_run:
        return 0

    if shutil.which("node") is None:
        print("error: node is required on PATH. Run ./scripts/bootstrap-node-deps.sh first.", file=sys.stderr)
        return 1

    version = node_version()
    if not _is_supported_node_version(version):
        print(f"error: Node.js 22+ is required; found {version or 'no node installation'}.", file=sys.stderr)
        return 1

    if shutil.which("npm") is None:
        print("error: npm is required on PATH. Run ./scripts/bootstrap-node-deps.sh first.", file=sys.stderr)
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
            f"error: {node_modules} is missing. Run ./scripts/bootstrap-node-deps.sh first.",
            file=sys.stderr,
        )
        return 1
    if callable_agents and not server_config.no_security:
        print(
            "error: callable local agents require no_security: true so they can rename without stored credentials.",
            file=sys.stderr,
        )
        return 1

    server_process = subprocess.Popen(
        command,
        cwd=server_config.showdown_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    server_log_threads = _start_prefixed_log_threads(server_process, "server")
    viewer_process: Optional[subprocess.Popen[str]] = None
    viewer_log_threads: list[threading.Thread] = []
    agent_processes: list[tuple[str, subprocess.Popen[str], list[threading.Thread]]] = []

    try:
        _wait_for_server_ready(server_config, server_process)
        if server_config.transcript_viewer.enabled:
            viewer_command = [
                sys.executable,
                "-m",
                "pokerena",
                "server",
                "transcript-viewer",
                "--config",
                args.config,
            ]
            viewer_process = subprocess.Popen(
                viewer_command,
                cwd=project_root,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            viewer_log_threads = _start_prefixed_log_threads(viewer_process, "viewer")
            _wait_for_http_ready(transcript_viewer_url(server_config), viewer_process, "transcript viewer")

        for agent in callable_agents:
            agent_command = [
                sys.executable,
                "-m",
                "pokerena",
                "agent",
                "showdown-client",
                "--config",
                args.config,
                "--agents-config",
                args.agents_config,
                "--agent-id",
                agent.agent_id,
            ]
            process = subprocess.Popen(
                agent_command,
                cwd=project_root,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            agent_processes.append(
                (agent.agent_id, process, _start_prefixed_log_threads(process, f"agent:{agent.agent_id}"))
            )

        while True:
            server_code = server_process.poll()
            if server_code is not None:
                return server_code
            if viewer_process is not None:
                viewer_code = viewer_process.poll()
                if viewer_code is not None:
                    print(
                        f"error: transcript viewer exited unexpectedly with code {viewer_code}.",
                        file=sys.stderr,
                    )
                    return viewer_code or 1
            for agent_id, process, _ in agent_processes:
                code = process.poll()
                if code is not None:
                    print(
                        f"error: callable agent {agent_id!r} exited unexpectedly with code {code}.",
                        file=sys.stderr,
                    )
                    return code or 1
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Shutting down Pokerena local server...")
        return 130
    finally:
        for _, process, threads in agent_processes:
            _terminate_process(process)
            _join_threads(threads)
        if viewer_process is not None:
            _terminate_process(viewer_process)
            _join_threads(viewer_log_threads)
        _terminate_process(server_process)
        _join_threads(server_log_threads)


def run_calc_damage(args: argparse.Namespace) -> int:
    project_root = detect_project_root()
    payload = read_damage_calc_input(
        input_path=args.input,
        use_stdin=args.stdin,
    )
    result = run_damage_calc(
        payload,
        project_root=project_root,
        timeout_seconds=args.timeout,
    )
    print(json.dumps(result, indent=2))
    return 0


def run_calc_damage_batch(args: argparse.Namespace) -> int:
    project_root = detect_project_root()
    payload = read_damage_calc_batch_input(
        input_path=args.input,
        use_stdin=args.stdin,
    )
    result = run_damage_calc_batch(
        payload,
        project_root=project_root,
        timeout_seconds=args.timeout,
    )
    print(json.dumps(result, indent=2))
    return 0


def run_agent_context(args: argparse.Namespace) -> int:
    context, _, _, _ = _load_replay_context(args)
    print(json.dumps(asdict(context), indent=2))
    return 0


def run_agent_decide(args: argparse.Namespace) -> int:
    context, agent, session, cursor_path = _load_replay_context(args)
    runtime_root = Path.cwd() / ".runtime"
    capture_path = Path(args.capture).resolve()
    decision_timeout = _resolve_decision_timeout(agent, args.decision_timeout)
    decision, artifacts = invoke_agent(
        agent=agent,
        context=context,
        runtime_root=runtime_root,
        capture_path=capture_path,
        dry_run=args.dry_run,
        timeout_seconds=decision_timeout,
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
    transcript_path = default_transcript_path(runtime_root=runtime_root, agent=agent, battle_id=battle_id)
    cursor = load_cursor(cursor_path)
    policy = DecisionPolicy(
        max_invalid_retries=args.max_invalid_retries,
        decision_timeout_seconds=_resolve_decision_timeout(agent, args.decision_timeout),
        history_limit=args.history_limit,
    )
    session = BattleSession(
        battle_id=battle_id,
        player_slot=agent.player_slot,
        history_limit=policy.history_limit,
    )
    pending_validation_request_sequence: Optional[int] = None
    update_transcript_metadata(
        transcript_path,
        agent=agent,
        battle_id=battle_id,
        format_name=format_id,
        finished=False,
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
            pending_validation_request_sequence = _update_submission_validation(
                event=event,
                session=session,
                transcript_path=transcript_path,
                pending_request_sequence=pending_validation_request_sequence,
            )

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
                        transcript_path=transcript_path,
                        policy=policy,
                    )
                    _submit_choice(
                        adapter=adapter,
                        session=session,
                        player_slot=agent.player_slot,
                        choice=decision_text,
                        rqid=session.current_rqid,
                    )
                    _mark_transcript_submitted(
                        transcript_path=transcript_path,
                        context=context,
                        choice=decision_text,
                    )
                    pending_validation_request_sequence = context.request_sequence
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
                if pending_validation_request_sequence is not None:
                    update_transcript_entry_state(
                        transcript_path,
                        request_sequence=pending_validation_request_sequence,
                        submission_state="accepted",
                        submission_detail="Battle finished after the decision was processed.",
                    )
                    pending_validation_request_sequence = None
                update_transcript_metadata(
                    transcript_path,
                    agent=agent,
                    battle_id=battle_id,
                    format_name=session.format_name,
                    winner=event.payload.get("winner"),
                    finished=True,
                )
                upsert_battle_summary_entry(transcript_path)
                winner = event.payload.get("winner")
                if winner:
                    print(f"Battle finished: winner={winner}")
                print(f"Capture: {capture_path}")
                return 0
    finally:
        save_capture(capture_path, session.to_capture())
        update_transcript_metadata(
            transcript_path,
            agent=agent,
            battle_id=battle_id,
            format_name=session.format_name,
            finished=session.finished,
        )
        adapter.close()


def run_agent_showdown_client(args: argparse.Namespace) -> int:
    project_root = Path.cwd()
    server_config = load_server_config(config_path=args.config, project_root=project_root)
    agents = load_agents_config(config_path=args.agents_config, project_root=project_root)
    agent = find_agent(agents, args.agent_id)
    if not agent.enabled:
        raise ConfigError(f"Agent {agent.agent_id!r} is disabled in the agents config.")
    if agent.transport != "showdown-client":
        raise ConfigError(
            f"Agent {agent.agent_id!r} uses transport {agent.transport!r}. Use transport showdown-client for callable local battles."
        )
    if not agent.callable.enabled:
        raise ConfigError(f"Agent {agent.agent_id!r} is not marked callable in the agents config.")

    runtime_root = project_root / ".runtime"
    policy = DecisionPolicy(
        max_invalid_retries=args.max_invalid_retries,
        decision_timeout_seconds=_resolve_decision_timeout(agent, args.decision_timeout),
        history_limit=args.history_limit,
    )
    adapter = ShowdownClientAdapter(server_config=server_config, agent=agent)

    session: Optional[BattleSession] = None
    cursor = AgentContextCursor(last_turn_number=None, last_request_sequence=0)
    cursor_path: Optional[Path] = None
    capture_path: Optional[Path] = None
    transcript_path: Optional[Path] = None
    challenger: Optional[str] = None
    pending_validation_request_sequence: Optional[int] = None

    try:
        adapter.connect()
        print(
            f"Callable agent {agent.agent_id} is online as {agent.callable.username} at {adapter.websocket_url()}"
        )
        while True:
            if session is not None and _battle_stop_is_active(runtime_root, agent, session.battle_id):
                _handle_manual_stop_request(
                    runtime_root=runtime_root,
                    adapter=adapter,
                    agent=agent,
                    session=session,
                    transcript_path=transcript_path,
                )
            try:
                event = adapter.next_event(timeout=0.25)
            except TimeoutError:
                continue

            if event.event_type == "client_notice":
                print(f"[notice] {event.payload.get('message', '')}")
                continue

            if event.event_type == "battle_started":
                session = BattleSession(
                    battle_id=event.battle_id,
                    player_slot=agent.player_slot,
                    history_limit=policy.history_limit,
                )
                capture_path = default_capture_path(runtime_root=runtime_root, agent=agent, battle_id=event.battle_id)
                cursor_path = default_cursor_path(runtime_root=runtime_root, agent=agent, battle_id=event.battle_id)
                transcript_path = default_transcript_path(runtime_root=runtime_root, agent=agent, battle_id=event.battle_id)
                cursor = load_cursor(cursor_path)
                challenger = event.payload.get("challenger") or None
                update_transcript_metadata(
                    transcript_path,
                    agent=agent,
                    battle_id=event.battle_id,
                    format_name=event.payload.get("format"),
                    challenger=challenger,
                    finished=False,
                )
                print(
                    f"Battle started: {event.battle_id} challenger={event.payload.get('challenger') or 'unknown'} "
                    f"format={event.payload.get('format') or 'unknown'}"
                )
                continue

            if session is None or event.battle_id != session.battle_id:
                continue

            session.ingest(event)
            if transcript_path is not None:
                pending_validation_request_sequence = _update_submission_validation(
                    event=event,
                    session=session,
                    transcript_path=transcript_path,
                    pending_request_sequence=pending_validation_request_sequence,
                )
            if capture_path is not None:
                save_capture(capture_path, session.to_capture())

            if event.event_type == "request_received" and session.waiting_for_decision:
                if _battle_stop_is_active(runtime_root, agent, session.battle_id):
                    continue
                if capture_path is None or cursor_path is None or transcript_path is None:
                    raise ConfigError("Showdown-client runtime is missing capture or cursor paths.")
                context = session.build_turn_context(agent=agent, cursor=cursor)
                try:
                    decision_text = _decide_or_fallback(
                        agent=agent,
                        session=session,
                        context=context,
                        runtime_root=runtime_root,
                        capture_path=capture_path,
                        transcript_path=transcript_path,
                        challenger=challenger,
                        policy=policy,
                    )
                except ManualBattleStopRequested:
                    _handle_manual_stop_request(
                        runtime_root=runtime_root,
                        adapter=adapter,
                        agent=agent,
                        session=session,
                        transcript_path=transcript_path,
                        context=context,
                    )
                    continue
                _submit_choice(
                    adapter=adapter,
                    session=session,
                    player_slot=agent.player_slot,
                    choice=decision_text,
                    rqid=session.current_rqid,
                )
                _mark_transcript_submitted(
                    transcript_path=transcript_path,
                    context=context,
                    choice=decision_text,
                )
                pending_validation_request_sequence = context.request_sequence
                cursor = session.advance_cursor()
                save_cursor(cursor_path, cursor)
                save_capture(capture_path, session.to_capture())
                continue

            if event.event_type == "battle_finished":
                if transcript_path is not None and pending_validation_request_sequence is not None:
                    update_transcript_entry_state(
                        transcript_path,
                        request_sequence=pending_validation_request_sequence,
                        submission_state="accepted",
                        submission_detail="Battle finished after the decision was processed.",
                    )
                    pending_validation_request_sequence = None
                if transcript_path is not None:
                    update_transcript_metadata(
                        transcript_path,
                        agent=agent,
                        battle_id=event.battle_id,
                        format_name=session.format_name if session is not None else None,
                        challenger=challenger,
                        winner=event.payload.get("winner"),
                        finished=True,
                    )
                    upsert_battle_summary_entry(transcript_path)
                clear_battle_stop_request(runtime_root, agent.agent_id, event.battle_id)
                winner = event.payload.get("winner")
                if winner:
                    print(f"Battle finished: winner={winner}")
                else:
                    print("Battle finished: tie")
                session = None
                capture_path = None
                cursor_path = None
                transcript_path = None
                challenger = None
                cursor = AgentContextCursor(last_turn_number=None, last_request_sequence=0)
    finally:
        if session is not None and capture_path is not None:
            save_capture(capture_path, session.to_capture())
        if session is not None and transcript_path is not None:
            update_transcript_metadata(
                transcript_path,
                agent=agent,
                battle_id=session.battle_id,
                format_name=session.format_name,
                challenger=challenger,
                finished=session.finished,
            )
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
    transcript_path: Path,
    challenger: Optional[str] = None,
    policy: DecisionPolicy,
) -> str:
    prompt_text = render_turn_prompt(agent, context)
    _record_transcript_start(
        transcript_path=transcript_path,
        agent=agent,
        session=session,
        context=context,
        prompt_text=prompt_text,
        challenger=challenger,
        timeout_seconds=policy.decision_timeout_seconds,
    )
    _append_turn_trace(
        transcript_path=transcript_path,
        context=context,
        kind="status",
        message="Prompt built and agent turn started.",
    )
    if session.current_invalid_attempts() >= policy.max_invalid_retries:
        fallback_decision = choose_first_legal(session.current_request)
        _update_transcript_result(
            transcript_path=transcript_path,
            context=context,
            decision=_fallback_decision_payload(fallback_decision, "max invalid retries reached"),
            usage=None,
            fallback_used=True,
            fallback_reason="max-invalid-retries",
            error=context.last_error,
            selected_action=fallback_decision,
            selected_action_label=_selected_action_label(fallback_decision, context.request),
            selected_action_source="fallback-first-legal",
        )
        _append_turn_trace(
            transcript_path=transcript_path,
            context=context,
            kind="status",
            message=f"Max invalid retries reached. Falling back to first legal action: {fallback_decision}.",
        )
        return fallback_decision

    try:
        decision, artifacts = invoke_agent(
            agent=agent,
            context=context,
            runtime_root=runtime_root,
            capture_path=capture_path,
            dry_run=False,
            timeout_seconds=policy.decision_timeout_seconds,
            trace_sink=lambda kind, message, **trace_kwargs: _append_turn_trace(
                transcript_path=transcript_path,
                context=context,
                kind=kind,
                message=message,
                actor_kind=trace_kwargs.get("actor_kind"),
                actor_name=trace_kwargs.get("actor_name"),
            ),
            cancel_check=lambda: _battle_stop_cancel_reason(runtime_root, agent, session.battle_id),
        )
    except AgentCancelledError as error:
        _update_transcript_result(
            transcript_path=transcript_path,
            context=context,
            decision=None,
            usage=None,
            fallback_used=False,
            fallback_reason="manual-stop",
            error=str(error),
            selected_action="/forfeit",
            selected_action_label=None,
            selected_action_source="manual-stop",
            turn_state="stopped",
        )
        _append_turn_trace(
            transcript_path=transcript_path,
            context=context,
            kind="status",
            message="Manual stop requested. Cancelling the agent turn before sending /forfeit.",
        )
        raise ManualBattleStopRequested(str(error)) from error
    except AgentTimeoutError as error:
        fallback_decision = choose_random_legal(session.current_request)
        _update_transcript_result(
            transcript_path=transcript_path,
            context=context,
            decision=None,
            usage=None,
            fallback_used=True,
            fallback_reason="timeout",
            error=str(error),
            selected_action=fallback_decision,
            selected_action_label=_selected_action_label(fallback_decision, context.request),
            selected_action_source="timeout-random",
            turn_state="timed_out",
            timed_out_at=_timestamp(),
        )
        _append_turn_trace(
            transcript_path=transcript_path,
            context=context,
            kind="status",
            message=f"Timed out after {policy.decision_timeout_seconds}s. Random legal fallback selected: {fallback_decision}.",
        )
        return fallback_decision
    except ConfigError as error:
        fallback_decision = choose_first_legal(session.current_request)
        _update_transcript_result(
            transcript_path=transcript_path,
            context=context,
            decision=None,
            usage=None,
            fallback_used=True,
            fallback_reason="agent-error",
            error=str(error),
            selected_action=fallback_decision,
            selected_action_label=_selected_action_label(fallback_decision, context.request),
            selected_action_source="fallback-first-legal",
        )
        _append_turn_trace(
            transcript_path=transcript_path,
            context=context,
            kind="status",
            message=f"Agent error. Falling back to first legal action: {fallback_decision}.",
        )
        return fallback_decision

    if decision is None:
        fallback_decision = choose_first_legal(session.current_request)
        _update_transcript_result(
            transcript_path=transcript_path,
            context=context,
            decision=None,
            usage=None,
            fallback_used=True,
            fallback_reason="empty-decision",
            error="Agent invocation returned no decision.",
            selected_action=fallback_decision,
            selected_action_label=_selected_action_label(fallback_decision, context.request),
            selected_action_source="fallback-first-legal",
        )
        _append_turn_trace(
            transcript_path=transcript_path,
            context=context,
            kind="status",
            message=f"Agent returned no decision. Falling back to first legal action: {fallback_decision}.",
        )
        return fallback_decision
    _update_transcript_result(
        transcript_path=transcript_path,
        context=context,
        decision=decision,
        usage=artifacts.usage,
        fallback_used=False,
        fallback_reason=None,
        error=None,
        selected_action=decision.decision,
        selected_action_label=_selected_action_label(decision.decision, context.request),
        selected_action_source="agent",
    )
    _append_turn_trace(
        transcript_path=transcript_path,
        context=context,
        kind="status",
        message=f"Decision parsed: {decision.decision}.",
    )
    return decision.decision


def _record_transcript_start(
    *,
    transcript_path: Path,
    agent,
    session: BattleSession,
    context,
    prompt_text: str,
    challenger: Optional[str],
    timeout_seconds: int,
) -> None:
    record_transcript_entry(
        transcript_path,
        agent=agent,
        battle_id=session.battle_id,
        format_name=session.format_name,
        challenger=challenger,
        winner=None,
        finished=session.finished,
        entry=TranscriptEntry(
            turn_number=context.turn_number,
            request_sequence=context.request_sequence,
            request_kind=context.request_kind,
            rqid=context.rqid,
            decision_attempt=context.decision_attempt,
            prompt_text=prompt_text,
            recent_public_events=list(context.recent_public_events),
            decision=None,
            raw_output="",
            notes="",
            fallback_used=False,
            fallback_reason=None,
            error=None,
            entry_kind="turn",
            turn_state="thinking",
            submission_state="pending",
            submission_detail=None,
            selected_action=None,
            selected_action_source=None,
            timeout_seconds=timeout_seconds,
            trace_events=[],
            decision_latency_ms=None,
            usage=None,
            summary=None,
            submitted_at=None,
            timed_out_at=None,
            validated_at=None,
            created_at=_timestamp(),
        ),
    )


def _update_transcript_result(
    *,
    transcript_path: Path,
    context,
    decision: Optional[AgentDecision],
    usage: Optional[dict],
    fallback_used: bool,
    fallback_reason: Optional[str],
    error: Optional[str],
    selected_action: Optional[str],
    selected_action_label: Optional[str],
    selected_action_source: Optional[str],
    turn_state: Optional[str] = None,
    timed_out_at: Optional[str] = None,
) -> None:
    safe_usage = _sanitize_usage_payload(usage)
    update_transcript_entry(
        transcript_path,
        request_sequence=context.request_sequence,
        decision_attempt=context.decision_attempt,
        decision=decision.decision if decision is not None else None,
        raw_output=decision.raw_output if decision is not None else "",
        notes=decision.notes if decision is not None else "",
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        error=error,
        usage=safe_usage,
        selected_action=selected_action,
        selected_action_label=selected_action_label,
        selected_action_source=selected_action_source,
        turn_state=turn_state or "thinking",
        timed_out_at=timed_out_at,
    )


def _fallback_decision_payload(decision_text: str, notes: str) -> AgentDecision:
    return AgentDecision(
        schema_version="pokerena.decision.v1",
        decision=decision_text,
        notes=notes,
        raw_output=decision_text,
    )


def _selected_action_label(choice: Optional[str], request: Optional[dict]) -> Optional[str]:
    if not choice or not isinstance(request, dict):
        return None

    action, _, remainder = choice.partition(" ")
    if not remainder:
        return None

    if action == "move":
        active = request.get("active")
        if not isinstance(active, list) or len(active) != 1 or not isinstance(active[0], dict):
            return None
        move_index = _parse_choice_index(remainder)
        if move_index is None:
            return None
        moves = active[0].get("moves")
        if not isinstance(moves, list) or move_index < 1 or move_index > len(moves):
            return None
        move = moves[move_index - 1]
        if not isinstance(move, dict):
            return None
        move_name = _resolve_move_name(move)
        if not move_name:
            return None
        return f'Used "{move_name}"'

    if action == "switch":
        side = request.get("side")
        if not isinstance(side, dict):
            return None
        pokemon = side.get("pokemon")
        if not isinstance(pokemon, list):
            return None
        switch_index = _parse_choice_index(remainder)
        if switch_index is None or switch_index < 1 or switch_index > len(pokemon):
            return None
        switch_target = pokemon[switch_index - 1]
        if not isinstance(switch_target, dict):
            return None
        pokemon_name = _resolve_switch_target_name(switch_target)
        if not pokemon_name:
            return None
        return f'Switch to "{pokemon_name}"'

    return None


def _parse_choice_index(value: str) -> Optional[int]:
    index_token = value.strip().split(maxsplit=1)[0]
    if not index_token.isdigit():
        return None
    index = int(index_token)
    return index if index > 0 else None


def _resolve_move_name(move: dict) -> Optional[str]:
    for key in ("move", "id"):
        value = move.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_switch_target_name(pokemon: dict) -> Optional[str]:
    details = pokemon.get("details")
    if isinstance(details, str) and details.strip():
        name = details.split(",", 1)[0].strip()
        if name:
            return name

    ident = pokemon.get("ident")
    if isinstance(ident, str) and ident.strip():
        _, _, name = ident.partition(":")
        resolved = name.strip()
        if resolved:
            return resolved

    return None


def _append_turn_trace(
    *,
    transcript_path: Path,
    context,
    kind: str,
    message: str,
    actor_kind: Optional[str] = None,
    actor_name: Optional[str] = None,
) -> None:
    append_transcript_trace_event(
        transcript_path,
        request_sequence=context.request_sequence,
        decision_attempt=context.decision_attempt,
        kind=kind,
        message=message,
        actor_kind=actor_kind,
        actor_name=actor_name,
    )


def _mark_transcript_submitted(
    *,
    transcript_path: Path,
    context,
    choice: str,
) -> None:
    entry = load_transcript_entry(
        transcript_path,
        request_sequence=context.request_sequence,
        decision_attempt=context.decision_attempt,
    )
    decision_latency_ms = None
    if entry is not None and isinstance(entry.get("created_at"), str):
        created_at = _parse_timestamp(entry["created_at"])
        if created_at is not None:
            decision_latency_ms = int((datetime.now(UTC) - created_at).total_seconds() * 1000)
    update_transcript_entry(
        transcript_path,
        request_sequence=context.request_sequence,
        decision_attempt=context.decision_attempt,
        selected_action=choice,
        selected_action_label=_selected_action_label(choice, context.request),
        turn_state="submitted",
        submission_state="pending",
        decision_latency_ms=decision_latency_ms,
        submitted_at=_timestamp(),
    )
    _append_turn_trace(
        transcript_path=transcript_path,
        context=context,
        kind="status",
        message=f"Submitted choice to Showdown: {choice}.",
    )


def _update_submission_validation(
    *,
    event,
    session: BattleSession,
    transcript_path: Path,
    pending_request_sequence: Optional[int],
) -> Optional[int]:
    if pending_request_sequence is None:
        return None

    if event.event_type == "choice_rejected" and event.player_slot == session.player_slot:
        detail = str(event.payload.get("message") or "Choice rejected.")
        update_transcript_entry_state(
            transcript_path,
            request_sequence=pending_request_sequence,
            submission_state="rejected",
            submission_detail=detail,
        )
        append_transcript_trace_event(
            transcript_path,
            request_sequence=pending_request_sequence,
            kind="status",
            message=f"Showdown rejected the submitted choice: {detail}",
        )
        return None

    if (
        event.event_type == "request_received"
        and event.player_slot == session.player_slot
        and session.current_request_sequence > pending_request_sequence
    ):
        update_transcript_entry_state(
            transcript_path,
            request_sequence=pending_request_sequence,
            submission_state="accepted",
            submission_detail="Showdown advanced to the next decision point.",
        )
        append_transcript_trace_event(
            transcript_path,
            request_sequence=pending_request_sequence,
            kind="status",
            message="Showdown advanced to the next decision point.",
        )
        return None

    if event.event_type == "public_update" and _public_update_confirms_submission(event.payload.get("lines", [])):
        update_transcript_entry_state(
            transcript_path,
            request_sequence=pending_request_sequence,
            submission_state="accepted",
            submission_detail="Battle progress confirmed the submitted choice.",
        )
        append_transcript_trace_event(
            transcript_path,
            request_sequence=pending_request_sequence,
            kind="status",
            message="Battle progress confirmed the submitted choice.",
        )
        return None

    if event.event_type == "battle_finished":
        update_transcript_entry_state(
            transcript_path,
            request_sequence=pending_request_sequence,
            submission_state="accepted",
            submission_detail="Battle finished after the decision was processed.",
        )
        append_transcript_trace_event(
            transcript_path,
            request_sequence=pending_request_sequence,
            kind="status",
            message="Battle finished after the decision was processed.",
        )
        return None

    return pending_request_sequence


def _public_update_confirms_submission(lines: object) -> bool:
    if not isinstance(lines, list):
        return False
    ignored_prefixes = ("|c|", "|chat|", "|inactive|", "|inactiveoff|", "|raw|", "|html|", "|uhtml|", "|uhtmlchange|")
    for item in lines:
        if not isinstance(item, str):
            continue
        line = item.strip()
        if not line or line == "|" or line.startswith("|t:|"):
            continue
        if line.startswith(ignored_prefixes):
            continue
        return True
    return False


def _submit_choice(
    *,
    adapter,
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


def _resolve_decision_timeout(agent, explicit_timeout: Optional[int]) -> int:
    if explicit_timeout is not None:
        return explicit_timeout
    return agent.hook.decision_timeout_seconds


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


def _pending_battle_stop_request(runtime_root: Path, agent, battle_id: str) -> Optional[dict]:
    payload = load_battle_stop_request(runtime_root, agent.agent_id, battle_id)
    if not isinstance(payload, dict):
        return None
    if payload.get("action") != "forfeit" or not payload.get("requested_at"):
        return None
    return payload


def _battle_stop_is_active(runtime_root: Path, agent, battle_id: str) -> bool:
    return _pending_battle_stop_request(runtime_root, agent, battle_id) is not None


def _battle_stop_cancel_reason(runtime_root: Path, agent, battle_id: str) -> Optional[str]:
    payload = _pending_battle_stop_request(runtime_root, agent, battle_id)
    if payload is None or payload.get("handled_at"):
        return None
    return "Battle manually stopped from Battle Sessions."


def _handle_manual_stop_request(
    *,
    runtime_root: Path,
    adapter: ShowdownClientAdapter,
    agent,
    session: BattleSession,
    transcript_path: Optional[Path],
    context=None,
) -> bool:
    payload = _pending_battle_stop_request(runtime_root, agent, session.battle_id)
    if payload is None:
        return False
    if payload.get("handled_at"):
        return True

    adapter.forfeit_current_battle()
    mark_battle_stop_handled(runtime_root, agent.agent_id, session.battle_id)

    request_sequence = (
        context.request_sequence
        if context is not None
        else session.current_request_sequence
    )
    if transcript_path is not None and request_sequence > 0:
        entry = load_transcript_entry(transcript_path, request_sequence=request_sequence)
        if entry is not None:
            update_transcript_entry(
                transcript_path,
                request_sequence=request_sequence,
                submission_state="rejected",
                submission_detail="Manual stop requested. Sent /forfeit to Showdown.",
                turn_state="stopped",
                error="Battle manually stopped from Battle Sessions.",
                selected_action="/forfeit",
                selected_action_source="manual-stop",
                validated_at=_timestamp(),
            )
            append_transcript_trace_event(
                transcript_path,
                request_sequence=request_sequence,
                kind="status",
                message="Manual stop requested. Sent /forfeit to Showdown.",
            )
    return True


def _wait_for_server_ready(
    server_config,
    server_process: subprocess.Popen[str],
    timeout_seconds: float = 20.0,
) -> None:
    _wait_for_http_ready(server_config.public_origin, server_process, "local Showdown server", timeout_seconds)


def _wait_for_http_ready(
    url: str,
    process: subprocess.Popen[str],
    label: str,
    timeout_seconds: float = 20.0,
) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Optional[str] = None
    while time.time() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise ConfigError(f"{label.capitalize()} exited before startup completed (code {returncode}).")
        try:
            with urllib_request.urlopen(url, timeout=1.0) as response:
                if 200 <= response.status < 500:
                    return
        except urllib_error.URLError as error:
            last_error = str(error)
        time.sleep(0.25)
    raise ConfigError(
        f"Timed out waiting for the {label} at {url}."
        + (f" Last error: {last_error}" if last_error else "")
    )


def _start_prefixed_log_threads(
    process: subprocess.Popen[str],
    label: str,
) -> list[threading.Thread]:
    threads: list[threading.Thread] = []
    if process.stdout is not None:
        threads.append(_start_log_thread(process.stdout, sys.stdout, label))
    if process.stderr is not None:
        threads.append(_start_log_thread(process.stderr, sys.stderr, label))
    return threads


def _start_log_thread(source: IO[str], target: IO[str], label: str) -> threading.Thread:
    thread = threading.Thread(target=_stream_logs, args=(source, target, label), daemon=True)
    thread.start()
    return thread


def _stream_logs(source: IO[str], target: IO[str], label: str) -> None:
    try:
        for line in source:
            print(f"[{label}] {line.rstrip()}", file=target)
    finally:
        source.close()


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _join_threads(threads: Sequence[threading.Thread]) -> None:
    for thread in threads:
        thread.join(timeout=1)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _sanitize_usage_payload(usage: object) -> Optional[dict]:
    if not isinstance(usage, dict):
        return None
    sanitized: dict[str, int | str] = {}
    provider = usage.get("provider")
    if isinstance(provider, str) and provider.strip():
        sanitized["provider"] = provider.strip()
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        value = usage.get(key)
        if isinstance(value, int):
            sanitized[key] = value
    return sanitized or None


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
        if server_config.transcript_viewer.enabled:
            results.append(CheckResult("transcript-viewer", True, transcript_viewer_url(server_config)))
        else:
            results.append(CheckResult("transcript-viewer", True, "disabled"))
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

    calc_script = project_root / CALC_SCRIPT_PATH
    calc_dependency = project_root / CALC_DEPENDENCY_PATH
    results.append(
        CheckResult(
            name="calc-script",
            ok=calc_script.exists(),
            detail=str(calc_script) if calc_script.exists() else f"missing {calc_script}",
        )
    )
    results.append(
        CheckResult(
            name="calc-deps",
            ok=calc_dependency.exists(),
            detail=str(calc_dependency) if calc_dependency.exists() else f"missing {calc_dependency}",
        )
    )
    if node_binary is None:
        detail = "install Node.js 22+ and rerun ./scripts/bootstrap-node-deps.sh"
        ok = False
    elif not _is_supported_node_version(version):
        detail = f"upgrade Node.js to 22+ (found {version})"
        ok = False
    elif not calc_script.exists() or not calc_dependency.exists():
        detail = "install root node dependencies with ./scripts/bootstrap-node-deps.sh"
        ok = False
    else:
        try:
            smoke_result = run_damage_calc(
                sample_damage_calc_payload(),
                project_root=project_root,
                timeout_seconds=DEFAULT_CALC_TIMEOUT_SECONDS,
            )
            calc_range = smoke_result.get("range", {})
            detail = f"range {calc_range.get('min')}-{calc_range.get('max')}"
            ok = True
        except ConfigError as error:
            detail = str(error)
            ok = False
    results.append(
        CheckResult(
            name="calc-smoke",
            ok=ok,
            detail=detail,
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
