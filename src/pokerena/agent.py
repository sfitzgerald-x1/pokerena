from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from importlib import resources
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Deque, Dict, List, Optional, Protocol

from jsonschema import Draft202012Validator

from .config import AgentDefinition, ConfigError, ServerConfig, _parse_dotenv


TURN_CONTEXT_SCHEMA_VERSION = "pokerena.turn-context.v1"
DECISION_SCHEMA_VERSION = "pokerena.decision.v1"
CAPTURE_SCHEMA_VERSION = "pokerena.battle-capture.v1"


@dataclass(frozen=True)
class DecisionPolicy:
    max_invalid_retries: int = 3
    decision_timeout_seconds: int = 45
    history_limit: int = 60
    fallback_policy: str = "first-legal"


@dataclass(frozen=True)
class AgentContextCursor:
    last_turn_number: Optional[int]
    last_request_sequence: int


@dataclass(frozen=True)
class SessionEvent:
    event_type: str
    battle_id: str
    payload: Dict[str, Any]
    player_slot: Optional[str] = None


@dataclass(frozen=True)
class TurnContext:
    schema_version: str
    battle_id: str
    agent_id: str
    provider: str
    player_slot: str
    context_token: str
    format_name: Optional[str]
    turn_number: Optional[int]
    phase: str
    request_kind: str
    rqid: Optional[str]
    request_sequence: int
    decision_attempt: int
    signals: Dict[str, bool]
    legal_action_hints: List[str]
    request: Optional[Dict[str, Any]]
    side: Optional[Dict[str, Any]]
    active: List[Dict[str, Any]]
    recent_public_events: List[str]
    last_error: Optional[str]


@dataclass(frozen=True)
class AgentDecision:
    schema_version: str
    decision: str
    notes: str
    raw_output: str


@dataclass(frozen=True)
class AgentInvocationArtifacts:
    context_path: Path
    prompt_path: Path
    response_path: Optional[Path]
    capture_path: Optional[Path]
    cursor_path: Path


@dataclass(frozen=True)
class BattleCapture:
    schema_version: str
    battle_id: str
    events: List[SessionEvent]


class Adapter(Protocol):
    def connect(self) -> None: ...

    def next_event(self) -> SessionEvent: ...

    def submit_decision(self, *, player_slot: str, choice: str, rqid: Optional[str]) -> None: ...

    def close(self) -> None: ...


class BattleSession:
    def __init__(
        self,
        *,
        battle_id: str,
        player_slot: str,
        history_limit: int = 60,
        best_effort_reprime: bool = False,
    ) -> None:
        self.battle_id = battle_id
        self.player_slot = player_slot
        self.history_limit = history_limit
        self.best_effort_reprime = best_effort_reprime
        self.events: List[SessionEvent] = []
        self.public_history: Deque[str] = deque(maxlen=history_limit)
        self.format_name: Optional[str] = None
        self.turn_number: Optional[int] = None
        self.phase = "setup"
        self.finished = False
        self.current_request: Optional[Dict[str, Any]] = None
        self.current_request_kind = "idle"
        self.current_rqid: Optional[str] = None
        self.current_request_sequence = 0
        self.current_request_turn_number: Optional[int] = None
        self.last_completed_request_turn_number: Optional[int] = None
        self.last_error: Optional[str] = None
        self.invalid_attempts_by_rqid: Dict[str, int] = {}
        self.synthetic_request_counter = 0

    @property
    def waiting_for_decision(self) -> bool:
        return bool(self.current_request) and not self.finished and self.current_request_kind not in {"wait", "idle"}

    def ingest(self, event: SessionEvent) -> None:
        self.events.append(event)
        if event.event_type == "public_update":
            self._ingest_public_update(event.payload.get("lines", []))
            return
        if event.event_type == "request_received" and event.player_slot == self.player_slot:
            self._ingest_request(event.payload)
            return
        if event.event_type == "choice_submitted" and event.player_slot == self.player_slot:
            self.last_error = None
            return
        if event.event_type == "choice_rejected" and event.player_slot == self.player_slot:
            self._ingest_rejection(event.payload)
            return
        if event.event_type == "battle_finished":
            self.finished = True
            self.phase = "finished"

    def build_turn_context(
        self,
        *,
        agent: AgentDefinition,
        cursor: AgentContextCursor,
    ) -> TurnContext:
        request = self.current_request
        active = request.get("active", []) if isinstance(request, dict) and isinstance(request.get("active"), list) else []
        side = request.get("side") if isinstance(request, dict) and isinstance(request.get("side"), dict) else None
        context = TurnContext(
            schema_version=agent.hook.context_format,
            battle_id=self.battle_id,
            agent_id=agent.agent_id,
            provider=agent.provider,
            player_slot=agent.player_slot,
            context_token=f"{self.battle_id}:{agent.player_slot}:{self.current_rqid or 'idle'}:{self.current_request_sequence}",
            format_name=self.format_name,
            turn_number=self.turn_number,
            phase=self.phase,
            request_kind=self.current_request_kind,
            rqid=self.current_rqid,
            request_sequence=self.current_request_sequence,
            decision_attempt=self.invalid_attempts_by_rqid.get(self.current_rqid or "", 0) + 1,
            signals={
                "turn_started": bool(
                    self.current_request_turn_number is not None
                    and self.current_request_turn_number != cursor.last_turn_number
                ),
                "request_updated": self.current_request_sequence != cursor.last_request_sequence,
                "decision_required": self.waiting_for_decision,
                "best_effort_reprime": self.best_effort_reprime,
            },
            legal_action_hints=_legal_action_hints(request),
            request=request,
            side=side,
            active=[item for item in active if isinstance(item, dict)],
            recent_public_events=list(self.public_history),
            last_error=self.last_error,
        )
        return context

    def advance_cursor(self) -> AgentContextCursor:
        return AgentContextCursor(
            last_turn_number=self.current_request_turn_number or self.turn_number,
            last_request_sequence=self.current_request_sequence,
        )

    def current_invalid_attempts(self) -> int:
        return self.invalid_attempts_by_rqid.get(self.current_rqid or "", 0)

    def to_capture(self) -> BattleCapture:
        return BattleCapture(
            schema_version=CAPTURE_SCHEMA_VERSION,
            battle_id=self.battle_id,
            events=list(self.events),
        )

    def _ingest_public_update(self, lines: List[str]) -> None:
        for line in lines:
            if not line:
                continue
            self.public_history.append(line)
            if line.startswith("|tier|"):
                _, _, value = line.partition("|tier|")
                self.format_name = value or self.format_name
            elif line.startswith("|turn|"):
                _, _, value = line.partition("|turn|")
                try:
                    self.turn_number = int(value)
                except ValueError as error:
                    raise ConfigError(f"Invalid turn marker: {line}") from error
                self.phase = "turn"
            elif line == "|teampreview":
                self.phase = "team-preview"
            elif line == "|upkeep":
                self.phase = "upkeep"
            elif line.startswith("|win|") or line == "|tie":
                self.finished = True
                self.phase = "finished"

    def _ingest_request(self, payload: Dict[str, Any]) -> None:
        self.current_request = payload
        self.current_request_kind = determine_request_kind(payload, finished=self.finished)
        self.phase = determine_phase(self.current_request_kind, self.turn_number, finished=self.finished)
        self.last_error = None
        self.current_request_turn_number = self.turn_number
        if not self.waiting_for_decision:
            return

        raw_rqid = payload.get("rqid")
        if raw_rqid is not None:
            self.current_rqid = str(raw_rqid)
        elif payload.get("update") and self.current_rqid:
            pass
        else:
            self.synthetic_request_counter += 1
            self.current_rqid = f"sim-{self.synthetic_request_counter}"
        self.current_request_sequence += 1
        if not payload.get("update"):
            self.last_completed_request_turn_number = self.current_request_turn_number

    def _ingest_rejection(self, payload: Dict[str, Any]) -> None:
        message = str(payload.get("message") or "Choice rejected.")
        self.last_error = message
        if self.current_rqid is None:
            return
        self.invalid_attempts_by_rqid[self.current_rqid] = self.invalid_attempts_by_rqid.get(self.current_rqid, 0) + 1


class SimStreamAdapter:
    def __init__(
        self,
        *,
        server_config: ServerConfig,
        format_id: str,
        battle_id: str,
        player_names: Dict[str, str],
        seed: Optional[List[int]] = None,
    ) -> None:
        self.server_config = server_config
        self.format_id = format_id
        self.battle_id = battle_id
        self.player_names = player_names
        self.seed = seed
        self.process: Optional[subprocess.Popen[str]] = None
        self.pending_events: Deque[SessionEvent] = deque()

    def connect(self) -> None:
        launcher = self.server_config.showdown_path / "pokemon-showdown"
        if not launcher.exists():
            raise ConfigError(f"Showdown launcher not found at {launcher}.")

        self.process = subprocess.Popen(
            [str(launcher), "simulate-battle"],
            cwd=self.server_config.showdown_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        start_spec: Dict[str, Any] = {"formatid": self.format_id}
        if self.seed is not None:
            start_spec["seed"] = self.seed
        self._write_line(f">start {json.dumps(start_spec, separators=(',', ':'))}")
        for player_slot in ("p1", "p2"):
            player_spec = {"name": self.player_names[player_slot]}
            self._write_line(f">player {player_slot} {json.dumps(player_spec, separators=(',', ':'))}")

    def next_event(self) -> SessionEvent:
        while not self.pending_events:
            chunk = self._read_chunk()
            self.pending_events.extend(self._parse_chunk(chunk))
        return self.pending_events.popleft()

    def submit_decision(self, *, player_slot: str, choice: str, rqid: Optional[str]) -> None:
        _ = rqid
        self._write_line(f">{player_slot} {choice}")

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process.stdout and not self.process.stdout.closed:
            self.process.stdout.close()
        if self.process.stderr and not self.process.stderr.closed:
            self.process.stderr.close()

    def _write_line(self, line: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise ConfigError("SimStreamAdapter is not connected.")
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def _read_chunk(self) -> str:
        if self.process is None or self.process.stdout is None:
            raise ConfigError("SimStreamAdapter is not connected.")

        lines: List[str] = []
        while True:
            line = self.process.stdout.readline()
            if line == "":
                stderr = ""
                returncode = self.process.poll()
                if returncode is not None and self.process.stderr is not None:
                    stderr = self.process.stderr.read().strip()
                raise ConfigError(
                    f"Simulator stream ended unexpectedly for {self.battle_id}."
                    + (f" stderr: {stderr}" if stderr else "")
                )
            line = line.rstrip("\n")
            if not line:
                if lines:
                    return "\n".join(lines)
                continue
            lines.append(line)

    def _parse_chunk(self, chunk: str) -> List[SessionEvent]:
        if "\n" not in chunk:
            raise ConfigError(f"Malformed simulator chunk: {chunk}")
        message_type, _, remainder = chunk.partition("\n")
        if message_type == "update":
            public_lines = _extract_public_lines(remainder.splitlines())
            return [
                SessionEvent(
                    event_type="public_update",
                    battle_id=self.battle_id,
                    payload={"lines": public_lines},
                )
            ]
        if message_type == "sideupdate":
            player_slot, _, side_data = remainder.partition("\n")
            return _parse_sideupdate_chunk(self.battle_id, player_slot, side_data)
        if message_type == "end":
            payload = json.loads(remainder) if remainder.strip() else {}
            return [
                SessionEvent(
                    event_type="battle_finished",
                    battle_id=self.battle_id,
                    payload=payload if isinstance(payload, dict) else {"value": payload},
                )
            ]
        raise ConfigError(f"Unsupported simulator message type: {message_type}")


class ShowdownClientAdapter:
    def __init__(self) -> None:
        self.enabled = False

    def connect(self) -> None:
        raise ConfigError(
            "The showdown-client adapter is defined but not implemented yet. Use the local sim-stream transport for now."
        )

    def next_event(self) -> SessionEvent:
        raise ConfigError("The showdown-client adapter is not implemented yet.")

    def submit_decision(self, *, player_slot: str, choice: str, rqid: Optional[str]) -> None:
        raise ConfigError("The showdown-client adapter is not implemented yet.")

    def close(self) -> None:
        return


def find_agent(agents: List[AgentDefinition], agent_id: str) -> AgentDefinition:
    for agent in agents:
        if agent.agent_id == agent_id:
            return agent
    raise ConfigError(f"Agent {agent_id!r} was not found in the loaded agents config.")


def default_capture_path(runtime_root: Path, agent: AgentDefinition, battle_id: str) -> Path:
    return runtime_root / "agents" / agent.agent_id / battle_id / "capture.json"


def default_cursor_path(runtime_root: Path, agent: AgentDefinition, battle_id: str) -> Path:
    return runtime_root / "agents" / agent.agent_id / battle_id / "cursor.json"


def load_capture(path: Path) -> BattleCapture:
    if not path.exists():
        raise ConfigError(f"Battle capture not found at {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"Battle capture at {path} must contain a JSON object.")
    schema_version = str(payload.get("schema_version") or "")
    if schema_version != CAPTURE_SCHEMA_VERSION:
        raise ConfigError(f"Unsupported battle capture schema {schema_version!r} in {path}.")
    raw_events = payload.get("events", [])
    if not isinstance(raw_events, list):
        raise ConfigError(f"Battle capture events in {path} must be a list.")

    events: List[SessionEvent] = []
    for item in raw_events:
        if not isinstance(item, dict):
            raise ConfigError(f"Battle capture event in {path} must be a JSON object.")
        events.append(
            SessionEvent(
                event_type=str(item.get("event_type") or ""),
                battle_id=str(item.get("battle_id") or payload.get("battle_id") or "capture"),
                player_slot=item.get("player_slot"),
                payload=item.get("payload") if isinstance(item.get("payload"), dict) else {},
            )
        )
    return BattleCapture(
        schema_version=schema_version,
        battle_id=str(payload.get("battle_id") or "capture"),
        events=events,
    )


def save_capture(path: Path, capture: BattleCapture) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": capture.schema_version,
        "battle_id": capture.battle_id,
        "events": [
            {
                "event_type": event.event_type,
                "battle_id": event.battle_id,
                "player_slot": event.player_slot,
                "payload": event.payload,
            }
            for event in capture.events
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_cursor(path: Path) -> AgentContextCursor:
    if not path.exists():
        return AgentContextCursor(last_turn_number=None, last_request_sequence=0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"Agent cursor at {path} must contain a JSON object.")
    turn_number = payload.get("last_turn_number")
    if turn_number is not None and not isinstance(turn_number, int):
        raise ConfigError(f"Agent cursor at {path} has an invalid last_turn_number.")
    request_sequence = payload.get("last_request_sequence", 0)
    if not isinstance(request_sequence, int):
        raise ConfigError(f"Agent cursor at {path} has an invalid last_request_sequence.")
    return AgentContextCursor(
        last_turn_number=turn_number,
        last_request_sequence=request_sequence,
    )


def save_cursor(path: Path, cursor: AgentContextCursor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "last_turn_number": cursor.last_turn_number,
                "last_request_sequence": cursor.last_request_sequence,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_session_from_capture(
    *,
    capture: BattleCapture,
    player_slot: str,
    history_limit: int,
) -> BattleSession:
    session = BattleSession(
        battle_id=capture.battle_id,
        player_slot=player_slot,
        history_limit=history_limit,
        best_effort_reprime=True,
    )
    for event in capture.events:
        session.ingest(event)
    return session


def render_turn_prompt(agent: AgentDefinition, context: TurnContext) -> str:
    context_json = json.dumps(asdict(context), indent=2, sort_keys=True)
    return "\n".join(
        [
            "You are a Pokemon Showdown battle agent running inside Pokerena.",
            f"Provider label: {agent.provider}.",
            "Use the JSON turn context as the source of truth.",
            "Forward legality from the request payload itself; do not invent actions outside it.",
            "Return JSON only with this shape:",
            json.dumps(
                {
                    "schema_version": agent.hook.decision_format,
                    "decision": "move 1",
                    "notes": "short explanation",
                }
            ),
            "If no action is required, return `wait`.",
            "",
            "TURN CONTEXT JSON",
            context_json,
        ]
    )


def invoke_agent(
    *,
    agent: AgentDefinition,
    context: TurnContext,
    runtime_root: Path,
    capture_path: Optional[Path],
    dry_run: bool,
    timeout_seconds: int,
) -> tuple[Optional[AgentDecision], AgentInvocationArtifacts]:
    agent_runtime_dir = runtime_root / "agents" / agent.agent_id / context.battle_id
    agent_runtime_dir.mkdir(parents=True, exist_ok=True)

    context_path = agent_runtime_dir / "turn-context.json"
    prompt_path = agent_runtime_dir / "turn-prompt.txt"
    response_path = agent_runtime_dir / "turn-response.json"
    cursor_path = agent_runtime_dir / "cursor.json"

    context_payload = asdict(context)
    _validate_schema("turn-context.v1.json", context_payload)
    context_path.write_text(json.dumps(context_payload, indent=2) + "\n", encoding="utf-8")
    prompt_path.write_text(render_turn_prompt(agent, context), encoding="utf-8")

    artifacts = AgentInvocationArtifacts(
        context_path=context_path,
        prompt_path=prompt_path,
        response_path=None if dry_run else response_path,
        capture_path=capture_path,
        cursor_path=cursor_path,
    )
    if dry_run:
        return None, artifacts

    env = os.environ.copy()
    if agent.env_file and agent.env_file.exists():
        env.update(_parse_dotenv(agent.env_file))
    env["POKERENA_TURN_CONTEXT_PATH"] = str(context_path)
    env["POKERENA_TURN_PROMPT_PATH"] = str(prompt_path)
    if capture_path is not None:
        env["POKERENA_BATTLE_CAPTURE_PATH"] = str(capture_path)

    try:
        completed = subprocess.run(
            [agent.launch.command, *agent.launch.args],
            input=prompt_path.read_text(encoding="utf-8"),
            text=True,
            capture_output=True,
            cwd=agent.launch.cwd,
            env=env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise ConfigError(
            f"Agent hook command timed out after {timeout_seconds} seconds."
        ) from error

    if completed.returncode != 0:
        raise ConfigError(
            f"Agent hook command exited with code {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
        )

    decision = parse_decision_output(completed.stdout, agent.hook.decision_format)
    response_path.write_text(json.dumps(asdict(decision), indent=2) + "\n", encoding="utf-8")
    return decision, artifacts


def parse_decision_output(stdout: str, expected_schema: str) -> AgentDecision:
    payload = stdout.strip()
    if not payload:
        raise ConfigError("Agent hook returned no output.")

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        decision = AgentDecision(
            schema_version=expected_schema,
            decision=payload.splitlines()[0].strip(),
            notes="",
            raw_output=payload,
        )
        _validate_schema("decision.v1.json", asdict(decision))
        return decision

    if not isinstance(parsed, dict):
        raise ConfigError("Agent hook must return a JSON object or a plain-text decision string.")
    decision_text = parsed.get("decision")
    if not isinstance(decision_text, str) or not decision_text.strip():
        raise ConfigError("Agent hook JSON response must include a non-empty `decision` string.")

    decision = AgentDecision(
        schema_version=str(parsed.get("schema_version") or expected_schema),
        decision=decision_text.strip(),
        notes=str(parsed.get("notes") or ""),
        raw_output=payload,
    )
    _validate_schema("decision.v1.json", asdict(decision))
    return decision


def choose_first_legal(request: Optional[Dict[str, Any]]) -> str:
    if not request:
        return "wait"
    if request.get("wait"):
        return "wait"
    if request.get("teamPreview"):
        side = request.get("side")
        pokemon = side.get("pokemon", []) if isinstance(side, dict) else []
        order = "".join(str(index) for index in range(1, len(pokemon) + 1))
        return f"team {order}" if order else "wait"

    force_switch = request.get("forceSwitch")
    if isinstance(force_switch, list) and any(bool(slot) for slot in force_switch):
        side = request.get("side")
        pokemon = side.get("pokemon", []) if isinstance(side, dict) else []
        available = [
            index + 1
            for index, slot in enumerate(pokemon)
            if isinstance(slot, dict) and _can_switch_to(slot)
        ]
        if not available:
            return "pass"
        choices: List[str] = []
        used: set[int] = set()
        for needed in force_switch:
            if not needed:
                choices.append("pass")
                continue
            target = next((candidate for candidate in available if candidate not in used), None)
            if target is None:
                choices.append("pass")
                continue
            used.add(target)
            choices.append(f"switch {target}")
        return ", ".join(choices)

    active = request.get("active")
    if isinstance(active, list) and active:
        choices = []
        for slot in active:
            if not isinstance(slot, dict):
                choices.append("pass")
                continue
            move = _first_enabled_move(slot)
            if move is None:
                choices.append("pass")
                continue
            choices.append(move)
        return ", ".join(choices)
    return "wait"


def determine_request_kind(request: Optional[Dict[str, Any]], *, finished: bool) -> str:
    if finished:
        return "finished"
    if not request:
        return "idle"
    if request.get("wait"):
        return "wait"
    if request.get("teamPreview"):
        return "team-preview"
    force_switch = request.get("forceSwitch")
    if isinstance(force_switch, list) and any(bool(slot) for slot in force_switch):
        return "switch"
    active = request.get("active")
    if isinstance(active, list) and active:
        return "move"
    return "unknown"


def determine_phase(request_kind: str, turn_number: Optional[int], *, finished: bool) -> str:
    if finished:
        return "finished"
    if request_kind == "team-preview":
        return "team-preview"
    if request_kind == "switch":
        return "switch"
    if request_kind == "move":
        return "turn"
    if request_kind == "wait":
        return "wait"
    if turn_number is not None:
        return "turn"
    return "setup"


def _parse_sideupdate_chunk(battle_id: str, player_slot: str, side_data: str) -> List[SessionEvent]:
    events: List[SessionEvent] = []
    for raw_line in side_data.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("|request|"):
            payload = json.loads(line[len("|request|") :])
            if not isinstance(payload, dict):
                raise ConfigError(f"Invalid request payload in sideupdate for {battle_id}.")
            events.append(
                SessionEvent(
                    event_type="request_received",
                    battle_id=battle_id,
                    player_slot=player_slot,
                    payload=payload,
                )
            )
        elif line.startswith("|error|"):
            events.append(
                SessionEvent(
                    event_type="choice_rejected",
                    battle_id=battle_id,
                    player_slot=player_slot,
                    payload={"message": line[len("|error|") :]},
                )
            )
    return events


def _extract_public_lines(lines: List[str]) -> List[str]:
    public_lines: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("|split|"):
            if index + 2 >= len(lines):
                raise ConfigError("Malformed |split| section in simulator update.")
            public_lines.append(lines[index + 2])
            index += 3
            continue
        public_lines.append(line)
        index += 1
    return public_lines


def _legal_action_hints(request: Optional[Dict[str, Any]]) -> List[str]:
    if not request:
        return []
    if request.get("wait"):
        return ["wait"]
    if request.get("teamPreview"):
        side = request.get("side")
        pokemon = side.get("pokemon", []) if isinstance(side, dict) else []
        if pokemon:
            return [f"team {''.join(str(index) for index in range(1, len(pokemon) + 1))}"]
        return ["team 123456"]

    hints: List[str] = []
    force_switch = request.get("forceSwitch")
    if isinstance(force_switch, list) and any(bool(slot) for slot in force_switch):
        side = request.get("side")
        pokemon = side.get("pokemon", []) if isinstance(side, dict) else []
        for index, slot in enumerate(pokemon, start=1):
            if isinstance(slot, dict) and _can_switch_to(slot):
                hints.append(f"switch {index}")

    active = request.get("active")
    if isinstance(active, list) and len(active) == 1:
        for active_slot in active:
            if not isinstance(active_slot, dict):
                continue
            for move_index, move in enumerate(active_slot.get("moves", []), start=1):
                if isinstance(move, dict) and not move.get("disabled", False):
                    hints.append(f"move {move_index}")

    if not hints:
        hints.append("wait")
    return hints


def _first_enabled_move(active_request: Dict[str, Any]) -> Optional[str]:
    moves = active_request.get("moves", [])
    if not isinstance(moves, list):
        return None
    for index, move in enumerate(moves, start=1):
        if isinstance(move, dict) and not move.get("disabled", False):
            return f"move {index}"
    return None


def _can_switch_to(pokemon: Dict[str, Any]) -> bool:
    if pokemon.get("active"):
        return False
    condition = str(pokemon.get("condition") or "")
    return "fnt" not in condition


def _load_schema(name: str) -> Dict[str, Any]:
    data = resources.files("pokerena").joinpath("schemas").joinpath(name).read_text(encoding="utf-8")
    return json.loads(data)


def _validate_schema(name: str, payload: Dict[str, Any]) -> None:
    validator = Draft202012Validator(_load_schema(name))
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.absolute_path))
    if not errors:
        return
    message = "; ".join(error.message for error in errors[:3])
    raise ConfigError(f"{name} validation failed: {message}")
