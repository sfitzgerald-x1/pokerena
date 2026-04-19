from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from importlib import resources
import json
import os
from pathlib import Path
import queue
import random
import re
import subprocess
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol
import unicodedata
from urllib.parse import urlsplit, urlunsplit

from jsonschema import Draft202012Validator

from .calc import (
    CALC_BATCH_REQUEST_SCHEMA_VERSION,
    CALC_REQUEST_SCHEMA_VERSION,
    CALC_SUPPORT_SUPPORTED_DAMAGING,
    CALC_SUPPORT_SUPPORTED_NON_DAMAGING,
    CALC_SUPPORT_UNSUPPORTED,
    classify_move_support,
)
from .config import AgentDefinition, ConfigError, ServerConfig, _parse_dotenv


TURN_CONTEXT_SCHEMA_VERSION = "pokerena.turn-context.v1"
DECISION_SCHEMA_VERSION = "pokerena.decision.v1"
CAPTURE_SCHEMA_VERSION = "pokerena.battle-capture.v1"
HOOK_ENV_ALLOWLIST = {
    "HOME",
    "PATH",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TMP",
    "TEMP",
    "COLORTERM",
    "NO_COLOR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "CODEX_HOME",
}
HOOK_ENV_PREFIX_ALLOWLIST = (
    "ANTHROPIC_",
    "OPENAI_",
    "AWS_",
    "AZURE_OPENAI_",
    "CLAUDE_",
)


@dataclass(frozen=True)
class DecisionPolicy:
    max_invalid_retries: int = 3
    decision_timeout_seconds: int = 120
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


class AgentTimeoutError(ConfigError):
    """Raised when a hook subprocess times out before producing a decision."""


class AgentCancelledError(ConfigError):
    """Raised when Pokerena cancels a hook subprocess before it returns."""


@dataclass(frozen=True)
class AgentInvocationArtifacts:
    context_path: Path
    prompt_path: Path
    response_path: Optional[Path]
    capture_path: Optional[Path]
    cursor_path: Path
    usage: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None


@dataclass(frozen=True)
class HookProcessResult:
    stdout_text: str
    usage: Optional[Dict[str, Any]]
    duration_ms: Optional[int]


@dataclass(frozen=True)
class BattleCapture:
    schema_version: str
    battle_id: str
    events: List[SessionEvent]


@dataclass(frozen=True)
class PreparedTurnPrompt:
    text: str
    trace_messages: List[str]


@dataclass
class PublicTurnBlock:
    turn_number: Optional[int]
    lines: List[str]


class Adapter(Protocol):
    def connect(self) -> None: ...

    def next_event(self, timeout: Optional[float] = None) -> SessionEvent: ...

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
        self.public_turns: Deque[PublicTurnBlock] = deque(maxlen=max(1, history_limit + 1))
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
            recent_public_events=self._recent_public_events(agent.hook.history_turn_limit),
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
            if line.startswith("|tier|"):
                self._append_public_line(line, self.turn_number)
                _, _, value = line.partition("|tier|")
                self.format_name = value or self.format_name
            elif line.startswith("|turn|"):
                _, _, value = line.partition("|turn|")
                try:
                    next_turn = int(value)
                except ValueError as error:
                    raise ConfigError(f"Invalid turn marker: {line}") from error
                self.turn_number = next_turn
                self._append_public_line(line, next_turn)
                self.phase = "turn"
            elif line == "|teampreview":
                self._append_public_line(line, self.turn_number)
                self.phase = "team-preview"
            elif line == "|upkeep":
                self._append_public_line(line, self.turn_number)
                self.phase = "upkeep"
            elif line.startswith("|win|") or line == "|tie":
                self._append_public_line(line, self.turn_number)
                self.finished = True
                self.phase = "finished"
            else:
                self._append_public_line(line, self.turn_number)

    def _ingest_request(self, payload: Dict[str, Any]) -> None:
        self.current_request = payload
        self.current_request_kind = determine_request_kind(payload, finished=self.finished)
        self.phase = determine_phase(self.current_request_kind, self.turn_number, finished=self.finished)
        if not payload.get("update"):
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

    def _append_public_line(self, line: str, turn_number: Optional[int]) -> None:
        if not self.public_turns or self.public_turns[-1].turn_number != turn_number:
            self.public_turns.append(PublicTurnBlock(turn_number=turn_number, lines=[]))
        self.public_turns[-1].lines.append(line)

    def _recent_public_events(self, turn_limit: int) -> List[str]:
        if self.turn_number is None:
            return [line for block in self.public_turns for line in block.lines]

        lowest_turn = max(1, self.turn_number - turn_limit + 1)
        recent_lines: List[str] = []
        for block in self.public_turns:
            if block.turn_number is None:
                if lowest_turn == 1:
                    recent_lines.extend(block.lines)
                continue
            if block.turn_number >= lowest_turn:
                recent_lines.extend(block.lines)
        return recent_lines


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

    def next_event(self, timeout: Optional[float] = None) -> SessionEvent:
        _ = timeout
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
        lines = chunk.splitlines()
        recognized_types = {"update", "sideupdate", "end"}
        while lines and lines[0] not in recognized_types:
            lines.pop(0)
        if not lines:
            return []

        normalized_chunk = "\n".join(lines)
        if "\n" not in normalized_chunk:
            raise ConfigError(f"Malformed simulator chunk: {chunk}")
        message_type, _, remainder = normalized_chunk.partition("\n")
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
    def __init__(
        self,
        *,
        server_config: ServerConfig,
        agent: AgentDefinition,
    ) -> None:
        self.server_config = server_config
        self.agent = agent
        self.connection = None
        self.pending_events: Deque[SessionEvent] = deque()
        self.authenticated = False
        self.current_battle_id: Optional[str] = None
        self.pending_challenger: Optional[str] = None
        self.pending_format: Optional[str] = None

    def connect(self) -> None:
        if not self.agent.callable.enabled:
            raise ConfigError(
                f"Agent {self.agent.agent_id!r} is not callable. Set callable.enabled: true to expose it on the local server."
            )
        if not self.agent.callable.username:
            raise ConfigError(
                f"Agent {self.agent.agent_id!r} must set callable.username for the showdown-client transport."
            )
        if not self.server_config.no_security:
            raise ConfigError(
                "Callable local agents require no_security: true so the bot can rename without checked-in credentials."
            )

        try:
            from websockets.sync.client import connect as websocket_connect
        except ImportError as error:
            raise ConfigError(
                "The Python websockets dependency is missing. Reinstall the Pokerena package to enable showdown-client."
            ) from error

        self.connection = websocket_connect(self.websocket_url(), open_timeout=10, close_timeout=5)
        while not self.authenticated:
            payload = self._recv_text()
            self.pending_events.extend(self._consume_message(payload))

    def next_event(self, timeout: Optional[float] = None) -> SessionEvent:
        while not self.pending_events:
            self.pending_events.extend(self._consume_message(self._recv_text(timeout=timeout)))
        return self.pending_events.popleft()

    def submit_decision(self, *, player_slot: str, choice: str, rqid: Optional[str]) -> None:
        if player_slot != self.agent.player_slot:
            raise ConfigError(
                f"Showdown client only supports local decisions for {self.agent.player_slot}; got {player_slot}."
            )
        if not self.current_battle_id:
            raise ConfigError("No active battle room is available for /choose.")
        suffix = f"|{rqid}" if rqid else ""
        self._send(self.current_battle_id, f"/choose {choice}{suffix}")

    def forfeit_current_battle(self) -> None:
        if not self.current_battle_id:
            raise ConfigError("No active battle room is available for /forfeit.")
        self._send(self.current_battle_id, "/forfeit")

    def close(self) -> None:
        if self.connection is None:
            return
        self.connection.close()
        self.connection = None
        return

    def websocket_url(self) -> str:
        parts = urlsplit(self.server_config.public_origin)
        if parts.scheme not in {"http", "https", "ws", "wss"}:
            raise ConfigError(
                f"Unsupported public_origin scheme {parts.scheme!r}; expected http(s) for the local Showdown server."
            )
        scheme = "wss" if parts.scheme in {"https", "wss"} else "ws"
        path = parts.path.rstrip("/")
        websocket_path = f"{path}/showdown/websocket" if path else "/showdown/websocket"
        return urlunsplit((scheme, parts.netloc, websocket_path, "", ""))

    def _recv_text(self, timeout: Optional[float] = None) -> str:
        if self.connection is None:
            raise ConfigError("The showdown-client adapter is not connected.")
        try:
            payload = self.connection.recv(timeout=timeout)
        except TimeoutError:
            raise
        except Exception as error:
            raise ConfigError(f"Showdown websocket receive failed: {error}") from error
        if payload is None:
            raise ConfigError("Showdown websocket closed unexpectedly.")
        if not isinstance(payload, str):
            raise ConfigError("Showdown websocket returned a non-text frame.")
        return payload

    def _send(self, room_id: str, text: str) -> None:
        if self.connection is None:
            raise ConfigError("The showdown-client adapter is not connected.")
        try:
            self.connection.send(f"{room_id}|{text}")
        except Exception as error:
            raise ConfigError(f"Showdown websocket send failed: {error}") from error

    def _send_global(self, text: str) -> None:
        self._send("", text)

    def _consume_message(self, payload: str) -> List[SessionEvent]:
        events: List[SessionEvent] = []
        for room_id, lines in _split_protocol_message(payload):
            if room_id is None:
                events.extend(self._consume_global_lines(lines))
            elif room_id.startswith("battle-"):
                events.extend(self._consume_battle_room(room_id, lines))
        return events

    def _consume_global_lines(self, lines: List[str]) -> List[SessionEvent]:
        events: List[SessionEvent] = []
        for line in lines:
            if line.startswith("|challstr|"):
                self._send_global(f"/trn {self.agent.callable.username}")
                continue
            if line.startswith("|updateuser|"):
                self._handle_updateuser(line)
                continue
            if line.startswith("|nametaken|"):
                self._handle_nametaken(line)
                continue
            if line.startswith("|pm|"):
                self._handle_pm(line)
                continue
            if line.startswith("|updatechallenges|"):
                self._handle_updatechallenges(line)
                continue
            if line.startswith("|popup|"):
                events.append(
                    SessionEvent(
                        event_type="client_notice",
                        battle_id=self.current_battle_id or "global",
                        payload={"message": line[len("|popup|") :]},
                    )
                )
        return events

    def _handle_pm(self, line: str) -> None:
        parts = line.split("|", 4)
        if len(parts) < 5:
            return
        sender = _user_id(parts[2])
        message = parts[4]
        if not message.startswith("/challenge "):
            return
        challenge_details = message[len("/challenge ") :]
        challenge_format, _, _ = challenge_details.partition("|")
        if challenge_format:
            self._handle_incoming_challenge(sender, challenge_format)

    def _handle_updateuser(self, line: str) -> None:
        parts = line.split("|", 5)
        if len(parts) < 5:
            return
        user = parts[2]
        named = parts[3] == "1"
        avatar = parts[4] if len(parts) > 4 else ""
        if _user_id(user) != _user_id(self.agent.callable.username or ""):
            return
        if not named:
            raise ConfigError(
                f"Showdown reported {self.agent.callable.username!r} without a stable local rename."
            )
        self.authenticated = True
        if self.agent.callable.avatar and avatar != self.agent.callable.avatar:
            self._send_global(f"/avatar {self.agent.callable.avatar}")

    def _handle_nametaken(self, line: str) -> None:
        parts = line.split("|", 3)
        if len(parts) < 4:
            return
        username = parts[2]
        message = parts[3]
        if _user_id(username) == _user_id(self.agent.callable.username or ""):
            raise ConfigError(
                f"Callable bot username {self.agent.callable.username!r} is unavailable: {message}"
            )

    def _handle_updatechallenges(self, line: str) -> None:
        try:
            payload = json.loads(line[len("|updatechallenges|") :])
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        challenges = payload.get("challengesFrom", {})
        if not isinstance(challenges, dict):
            return
        if self.pending_challenger and self.pending_challenger not in challenges:
            self.pending_challenger = None
            self.pending_format = None
        for challenger, challenge_format in challenges.items():
            if not isinstance(challenger, str) or not isinstance(challenge_format, str):
                continue
            self._handle_incoming_challenge(challenger, challenge_format)

    def _handle_incoming_challenge(self, challenger: str, challenge_format: str) -> None:
        if self.agent.callable.challenge_policy != "accept-direct-challenges":
            self._reject_challenge(challenger, "This bot is not accepting challenges right now.")
            return
        normalized_format = challenge_format.strip().lower()
        accepted_formats = {item.lower() for item in self.agent.callable.accepted_formats}
        if self.current_battle_id is not None or (
            self.pending_challenger is not None and self.pending_challenger != challenger
        ):
            self._reject_challenge(challenger, "I'm already in another battle.")
            return
        if normalized_format not in accepted_formats:
            allowed = ", ".join(self.agent.callable.accepted_formats) or "none"
            self._reject_challenge(challenger, f"I only accept these formats: {allowed}.")
            return
        if self.pending_challenger == challenger and self.pending_format == normalized_format:
            return
        self._send_global("/utm null")
        self._send_global(f"/accept {challenger}")
        self.pending_challenger = challenger
        self.pending_format = normalized_format

    def _reject_challenge(self, challenger: str, message: str) -> None:
        self._send_global(f"/pm {challenger}, {message}")
        self._send_global(f"/reject {challenger}")

    def _consume_battle_room(self, room_id: str, lines: List[str]) -> List[SessionEvent]:
        if self.current_battle_id and room_id != self.current_battle_id:
            self._send(room_id, "/forfeit")
            return [
                SessionEvent(
                    event_type="client_notice",
                    battle_id=room_id,
                    payload={
                        "message": (
                            "Forfeited an unexpected second battle room because callable local agents only support one active battle at a time."
                        )
                    },
                )
            ]

        events: List[SessionEvent] = []
        if self.current_battle_id != room_id:
            self.current_battle_id = room_id
            events.append(
                SessionEvent(
                    event_type="battle_started",
                    battle_id=room_id,
                    payload={
                        "challenger": self.pending_challenger,
                        "format": self.pending_format,
                    },
                )
            )

        public_lines: List[str] = []
        request_events: List[SessionEvent] = []
        rejection_events: List[SessionEvent] = []
        battle_finished_payload: Optional[Dict[str, Any]] = None
        for line in lines:
            if line == "|deinit":
                continue
            if line.startswith("|request|"):
                payload = json.loads(line[len("|request|") :])
                if not isinstance(payload, dict):
                    raise ConfigError(f"Invalid request payload in battle room {room_id}.")
                request_events.append(
                    SessionEvent(
                        event_type="request_received",
                        battle_id=room_id,
                        player_slot=self.agent.player_slot,
                        payload=payload,
                    )
                )
                continue
            if line.startswith("|error|"):
                rejection_events.append(
                    SessionEvent(
                        event_type="choice_rejected",
                        battle_id=room_id,
                        player_slot=self.agent.player_slot,
                        payload={"message": line[len("|error|") :]},
                    )
                )
                continue
            public_lines.append(line)
            if line.startswith("|win|"):
                battle_finished_payload = {"winner": line[len("|win|") :]}
            elif line == "|tie":
                battle_finished_payload = {"tie": True}

        if public_lines:
            events.append(
                SessionEvent(
                    event_type="public_update",
                    battle_id=room_id,
                    payload={"lines": public_lines},
                )
            )
        events.extend(request_events)
        events.extend(rejection_events)
        if battle_finished_payload is not None:
            events.append(
                SessionEvent(
                    event_type="battle_finished",
                    battle_id=room_id,
                    payload=battle_finished_payload,
                )
            )
            self.current_battle_id = None
            self.pending_challenger = None
            self.pending_format = None
        return events


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


def prepare_turn_prompt(
    agent: AgentDefinition,
    context: TurnContext,
    *,
    project_root: Optional[Path] = None,
) -> PreparedTurnPrompt:
    context_json = json.dumps(asdict(context), indent=2, sort_keys=True)
    sections = [
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
    ]
    switch_section = _voluntary_switch_prompt_section(context)
    if switch_section:
        sections.extend(["", switch_section])
    calc_section, trace_messages = _damage_calc_section(context, project_root=project_root)
    if calc_section:
        sections.extend(["", calc_section])
    sections.extend(
        [
            "",
            "TURN CONTEXT JSON",
            context_json,
        ]
    )
    return PreparedTurnPrompt(text="\n".join(sections), trace_messages=trace_messages)


def render_turn_prompt(
    agent: AgentDefinition,
    context: TurnContext,
    *,
    project_root: Optional[Path] = None,
) -> str:
    return prepare_turn_prompt(agent, context, project_root=project_root).text


def _voluntary_switch_prompt_section(context: TurnContext) -> str:
    if context.request_kind != "move":
        return ""
    switch_hints = [hint for hint in context.legal_action_hints if isinstance(hint, str) and hint.startswith("switch ")]
    if not switch_hints:
        return ""
    return "\n".join(
        [
            "VOLUNTARY SWITCHING",
            "This is a normal move turn, and switching is also legal here.",
            "Before locking in a move, also consider whether pivoting to a teammate improves the board position.",
            "If the request payload shows that you are trapped, treat switching as unavailable.",
            f"Available switch commands: {', '.join(f'`{hint}`' for hint in switch_hints)}",
        ]
    )


def _damage_calc_section(
    context: TurnContext,
    project_root: Optional[Path] = None,
) -> tuple[str, List[str]]:
    if context.request_kind != "move":
        return "", []

    calc_plan = _build_damage_calc_batch_plan(context, project_root=project_root)
    if calc_plan is None:
        return "\n".join(
            [
                "DAMAGE CALC WORKFLOW",
                "Use `python3.14 -m pokerena calc damage-batch --stdin` when you have enough observable information to compare legal move candidates.",
                "For obviously non-damaging/status moves, reason heuristically instead of forcing a calc.",
            ]
        ), []

    move_labels = list(calc_plan["move_labels"])
    skipped_moves = list(calc_plan["skipped_moves"])
    batch_payload = calc_plan.get("batch_payload")
    lines = [
        "DAMAGE CALC WORKFLOW",
        "On this turn you may either attack or switch.",
        "If you plan to attack, run one Pokerena damage batch for the supported damaging move candidates below.",
    ]
    if skipped_moves:
        lines.extend(["", "Skip these moves in damage calc and reason heuristically:"])
        for skipped in skipped_moves:
            lines.append(f"- {skipped['label']} — {skipped['reason']}")
    if not batch_payload:
        lines.extend(
            [
                "",
                "No supported damaging move candidates remain for the calc tool on this turn.",
                "If attacking looks bad, consider whether switching is the better play.",
            ]
        )
        return "\n".join(lines), list(calc_plan["trace_messages"])

    lines.extend(
        [
            "For clearly non-damaging/status moves, reason heuristically instead of forcing a calc.",
            "Use this command exactly:",
            "`python3.14 -m pokerena calc damage-batch --stdin`",
            "",
            "Move order:",
        ]
    )
    for index, label in enumerate(move_labels, start=1):
        lines.append(f"{index}. {label}")
    lines.extend(
        [
            "",
            "Ready-to-run batch request:",
            "```json",
            json.dumps(batch_payload, indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines), list(calc_plan["trace_messages"])


def _build_damage_calc_batch_plan(
    context: TurnContext,
    *,
    project_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    generation = _extract_generation(context)
    attacker = _active_side_pokemon(context.side)
    defender = _active_opponent_from_public_history(context)
    active_request = context.active[0] if len(context.active) == 1 and isinstance(context.active[0], dict) else None
    if generation is None or attacker is None or defender is None or active_request is None:
        return None

    requests: List[Dict[str, Any]] = []
    move_labels: List[str] = []
    skipped_moves: List[Dict[str, str]] = []
    trace_messages: List[str] = []
    moves = active_request.get("moves", [])
    if not isinstance(moves, list):
        return None
    for move_index, move in enumerate(moves, start=1):
        if not isinstance(move, dict) or move.get("disabled", False):
            continue
        move_name = str(move.get("move") or "").strip()
        if not move_name:
            continue
        label = f"Move {move_index} · {move_name}"
        support = _classify_move_for_prompt(
            move=move,
            generation=generation,
            move_name=move_name,
            project_root=project_root,
        )
        if support["classification"] != CALC_SUPPORT_SUPPORTED_DAMAGING:
            skipped_moves.append(
                {
                    "label": label,
                    "reason": _calc_skip_reason_text(support["classification"]),
                }
            )
            trace_messages.append(
                f"Skipping calc preflight for {label}: {_calc_trace_reason_text(support)}."
            )
            continue
        requests.append(
            {
                "schema_version": CALC_REQUEST_SCHEMA_VERSION,
                "generation": generation,
                "attacker": attacker,
                "defender": defender,
                "move": {"name": move_name},
                "field": {},
            }
        )
        move_labels.append(label)
    return {
        "batch_payload": (
            {
                "schema_version": CALC_BATCH_REQUEST_SCHEMA_VERSION,
                "requests": requests,
            }
            if requests
            else None
        ),
        "move_labels": move_labels,
        "skipped_moves": skipped_moves,
        "trace_messages": trace_messages,
    }


def _classify_move_for_prompt(
    *,
    move: Dict[str, Any],
    generation: int,
    move_name: str,
    project_root: Optional[Path],
) -> Dict[str, str]:
    if project_root is not None:
        try:
            return classify_move_support(
                project_root=project_root,
                generation=generation,
                move_name=move_name,
            )
        except ConfigError:
            pass
    if _move_is_probably_non_damaging(move):
        return {
            "classification": CALC_SUPPORT_SUPPORTED_NON_DAMAGING,
            "source": "heuristic",
        }
    return {
        "classification": CALC_SUPPORT_SUPPORTED_DAMAGING,
        "source": "heuristic",
    }


def _move_is_probably_non_damaging(move: Dict[str, Any]) -> bool:
    category = move.get("category")
    if isinstance(category, str) and category.strip().lower() == "status":
        return True
    base_power = move.get("basePower")
    if isinstance(base_power, int):
        return base_power == 0
    return False


def _calc_skip_reason_text(classification: str) -> str:
    if classification == CALC_SUPPORT_SUPPORTED_NON_DAMAGING:
        return "non-damaging/status move; reason heuristically."
    return "unsupported by the local calc tool; do not retry this calc."


def _calc_trace_reason_text(support: Dict[str, str]) -> str:
    source = str(support.get("source") or "preflight")
    classification = support["classification"]
    if classification == CALC_SUPPORT_SUPPORTED_NON_DAMAGING:
        return f"classified as non-damaging/status via {source}"
    return f"classified as unsupported via {source}"


def _extract_generation(context: TurnContext) -> Optional[int]:
    if context.format_name:
        match = re.search(r"\[Gen (\d+)\]", context.format_name)
        if match:
            return int(match.group(1))
    for line in context.recent_public_events:
        if not isinstance(line, str) or not line.startswith("|gen|"):
            continue
        _, _, value = line.partition("|gen|")
        try:
            return int(value)
        except ValueError:
            continue
    return None


def _active_side_pokemon(side: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(side, dict):
        return None
    pokemon = side.get("pokemon", [])
    if not isinstance(pokemon, list):
        return None
    for candidate in pokemon:
        if not isinstance(candidate, dict) or not candidate.get("active"):
            continue
        species = _pokemon_species(candidate)
        if not species:
            return None
        payload: Dict[str, Any] = {"species": species}
        options: Dict[str, Any] = {}
        level = _pokemon_level(candidate)
        if level is not None:
            options["level"] = level
        stats = candidate.get("stats")
        if isinstance(stats, dict) and stats:
            options["stats"] = stats
        item = candidate.get("item")
        if isinstance(item, str) and item.strip():
            options["item"] = item.strip()
        ability = candidate.get("baseAbility")
        if isinstance(ability, str) and ability.strip():
            options["ability"] = ability.strip()
        if options:
            payload["options"] = options
        return payload
    return None


def _active_opponent_from_public_history(context: TurnContext) -> Optional[Dict[str, Any]]:
    opponent_prefix = "p2" if context.player_slot == "p1" else "p1"
    for line in reversed(context.recent_public_events):
        if not isinstance(line, str):
            continue
        parts = line.split("|")
        if len(parts) < 4 or parts[1] not in {"switch", "drag", "replace"}:
            continue
        ident = parts[2]
        if not isinstance(ident, str) or not ident.startswith(f"{opponent_prefix}"):
            continue
        details = parts[3] if len(parts) > 3 else ""
        species = _species_from_details(details) or _species_from_ident(ident)
        if not species:
            return None
        payload: Dict[str, Any] = {"species": species}
        options: Dict[str, Any] = {}
        level = _level_from_details(details)
        if level is not None:
            options["level"] = level
        if options:
            payload["options"] = options
        return payload
    return None


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
    if isinstance(details, str):
        return _level_from_details(details)
    return None


def _species_from_details(details: str) -> Optional[str]:
    value = details.split(",", 1)[0].strip()
    return value or None


def _level_from_details(details: str) -> Optional[int]:
    match = re.search(r"\bL(\d+)\b", details)
    if not match:
        return None
    return int(match.group(1))


def _species_from_ident(ident: str) -> Optional[str]:
    if ":" not in ident:
        return ident.strip() or None
    value = ident.split(":", 1)[1].strip()
    return value or None


def invoke_agent(
    *,
    agent: AgentDefinition,
    context: TurnContext,
    runtime_root: Path,
    capture_path: Optional[Path],
    dry_run: bool,
    timeout_seconds: int,
    prompt_text: Optional[str] = None,
    trace_sink: Optional[Callable[..., None]] = None,
    cancel_check: Optional[Callable[[], Optional[str]]] = None,
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
    resolved_prompt_text = prompt_text or render_turn_prompt(
        agent,
        context,
        project_root=runtime_root.parent,
    )
    prompt_path.write_text(resolved_prompt_text, encoding="utf-8")

    artifacts = AgentInvocationArtifacts(
        context_path=context_path,
        prompt_path=prompt_path,
        response_path=None if dry_run else response_path,
        capture_path=capture_path,
        cursor_path=cursor_path,
    )
    if dry_run:
        return None, artifacts

    env = _hook_base_env()
    if agent.env_file and agent.env_file.exists():
        env.update(_parse_dotenv(agent.env_file))
    env["POKERENA_TURN_CONTEXT_PATH"] = str(context_path)
    env["POKERENA_TURN_PROMPT_PATH"] = str(prompt_path)
    env["POKERENA_TRANSCRIPT_PATH"] = str(agent_runtime_dir / "transcript.json")
    env["POKERENA_REQUEST_SEQUENCE"] = str(context.request_sequence)
    env["POKERENA_DECISION_ATTEMPT"] = str(context.decision_attempt)
    if capture_path is not None:
        env["POKERENA_BATTLE_CAPTURE_PATH"] = str(capture_path)

    command, claude_streaming = _hook_command(agent)
    hook_result = _run_hook_process(
        command=command,
        cwd=agent.launch.cwd,
        env=env,
        prompt_text=prompt_path.read_text(encoding="utf-8"),
        timeout_seconds=timeout_seconds,
        expected_schema=agent.hook.decision_format,
        claude_streaming=claude_streaming,
        trace_sink=trace_sink,
        cancel_check=cancel_check,
    )

    decision = parse_decision_output(hook_result.stdout_text, agent.hook.decision_format)
    response_path.write_text(json.dumps(asdict(decision), indent=2) + "\n", encoding="utf-8")
    return decision, AgentInvocationArtifacts(
        context_path=artifacts.context_path,
        prompt_path=artifacts.prompt_path,
        response_path=artifacts.response_path,
        capture_path=artifacts.capture_path,
        cursor_path=artifacts.cursor_path,
        usage=hook_result.usage,
        duration_ms=hook_result.duration_ms,
    )


def _hook_command(agent: AgentDefinition) -> tuple[List[str], bool]:
    command = [agent.launch.command, *agent.launch.args]
    executable = Path(agent.launch.command).name.lower()
    if executable != "claude":
        return command, False

    filtered: List[str] = []
    skip_next = False
    for index, arg in enumerate(agent.launch.args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--output-format":
            skip_next = True
            continue
        if arg in {"--include-partial-messages", "--verbose"}:
            continue
        filtered.append(arg)
        if arg == "-p":
            continue
        if arg == "--print":
            continue
        if arg == "--model" and index + 1 < len(agent.launch.args):
            filtered.append(agent.launch.args[index + 1])
            skip_next = True
    if "-p" not in filtered and "--print" not in filtered:
        filtered.insert(0, "-p")
    filtered.extend(["--output-format", "stream-json", "--include-partial-messages", "--verbose"])
    return [agent.launch.command, *filtered], True


def _run_hook_process(
    *,
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    prompt_text: str,
    timeout_seconds: int,
    expected_schema: str,
    claude_streaming: bool,
    trace_sink: Optional[Callable[..., None]],
    cancel_check: Optional[Callable[[], Optional[str]]],
) -> HookProcessResult:
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
        bufsize=1,
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        raise ConfigError("Agent hook subprocess did not expose stdio pipes.")
    process.stdin.write(prompt_text)
    process.stdin.close()

    if trace_sink is not None:
        trace_sink("status", f"Hook started: {' '.join(command)}")

    event_queue: "queue.Queue[tuple[str, Optional[str]]]" = queue.Queue()
    reader_threads = [
        threading.Thread(
            target=_stream_reader,
            args=(process.stdout, "stdout", event_queue),
            daemon=True,
        ),
        threading.Thread(
            target=_stream_reader,
            args=(process.stderr, "stderr", event_queue),
            daemon=True,
        ),
    ]
    for thread in reader_threads:
        thread.start()

    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []
    stdout_done = False
    stderr_done = False
    parser = _ClaudeStreamParser(trace_sink=trace_sink) if claude_streaming else None
    deadline = time.monotonic() + timeout_seconds

    try:
        while True:
            if stdout_done and stderr_done and process.poll() is not None:
                break
            cancel_reason = cancel_check() if cancel_check is not None else None
            if cancel_reason and process.poll() is None:
                _terminate_process(process)
                raise AgentCancelledError(cancel_reason)
            if time.monotonic() >= deadline and process.poll() is None:
                _terminate_process(process)
                raise AgentTimeoutError(
                    f"Agent hook command timed out after {timeout_seconds} seconds."
                )
            try:
                source, line = event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                if source == "stdout":
                    stdout_done = True
                else:
                    stderr_done = True
                continue
            if source == "stdout":
                stdout_chunks.append(line)
                if parser is not None:
                    parser.consume(line.rstrip("\n"))
                elif trace_sink is not None and line.strip():
                    trace_sink("agent", line)
            else:
                stderr_chunks.append(line)
                if trace_sink is not None and line.strip():
                    trace_sink("status", f"stderr: {line.rstrip()}")
    finally:
        for thread in reader_threads:
            thread.join(timeout=1)

    return_code = process.wait()
    if process.stdout is not None and not process.stdout.closed:
        process.stdout.close()
    if process.stderr is not None and not process.stderr.closed:
        process.stderr.close()
    stderr_text = "".join(stderr_chunks).strip()
    stdout_text = "".join(stdout_chunks).strip()
    if return_code != 0:
        raise ConfigError(
            f"Agent hook command exited with code {return_code}: {stderr_text or stdout_text}"
        )
    if parser is not None:
        final_output = parser.final_output()
        if final_output:
            return HookProcessResult(
                stdout_text=final_output,
                usage=parser.usage,
                duration_ms=parser.duration_ms,
            )
    return HookProcessResult(stdout_text=stdout_text, usage=None, duration_ms=None)


def _stream_reader(
    stream: Any,
    source: str,
    output: "queue.Queue[tuple[str, Optional[str]]]",
) -> None:
    try:
        for line in stream:
            output.put((source, line))
    finally:
        output.put((source, None))


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


class _ClaudeStreamParser:
    def __init__(self, *, trace_sink: Optional[Callable[..., None]]) -> None:
        self.trace_sink = trace_sink
        self.partial_text: List[str] = []
        self.assistant_output = ""
        self.result_output = ""
        self.usage: Optional[Dict[str, Any]] = None
        self.duration_ms: Optional[int] = None

    def consume(self, line: str) -> None:
        if not line.strip():
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if self.trace_sink is not None:
                self.trace_sink("agent", line + "\n")
            return
        if not isinstance(payload, dict):
            return

        payload_type = payload.get("type")
        if payload_type == "stream_event":
            event = payload.get("event")
            if not isinstance(event, dict):
                return
            event_type = event.get("type")
            if event_type == "content_block_delta":
                delta = event.get("delta")
                if isinstance(delta, dict) and delta.get("type") == "text_delta":
                    text = str(delta.get("text") or "")
                    if text:
                        self.partial_text.append(text)
                        if self.trace_sink is not None:
                            self.trace_sink("agent", text)
                return
            if event_type == "content_block_start":
                content_block = event.get("content_block")
                if isinstance(content_block, dict) and content_block.get("type") == "tool_use":
                    tool_name = str(content_block.get("name") or "tool")
                    if self.trace_sink is not None:
                        self.trace_sink(
                            "status",
                            f"Claude using tool: {tool_name}",
                            actor_kind="helper",
                            actor_name=tool_name,
                        )
                return
            if event_type == "message_delta":
                self._update_usage(event.get("usage"))
                return
            return

        if payload_type == "assistant":
            message = payload.get("message")
            if not isinstance(message, dict):
                return
            self._update_usage(message.get("usage"))
            content = message.get("content")
            if not isinstance(content, list):
                return
            texts = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                text = str(block.get("text") or "")
                if text:
                    texts.append(text)
            if texts:
                self.assistant_output = "".join(texts)
            return

        if payload_type == "result":
            result = payload.get("result")
            if isinstance(result, str) and result.strip():
                self.result_output = result.strip()
            self._update_usage(payload.get("usage"))
            duration_ms = payload.get("duration_ms")
            if isinstance(duration_ms, int):
                self.duration_ms = duration_ms

    def final_output(self) -> str:
        for candidate in (self.result_output, self.assistant_output, "".join(self.partial_text).strip()):
            if candidate:
                return candidate
        return ""

    def _update_usage(self, usage: Any) -> None:
        normalized = _normalize_usage_payload(usage)
        if normalized is not None:
            self.usage = normalized


def _normalize_usage_payload(usage: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(usage, dict):
        return None
    input_tokens = _usage_int(usage.get("input_tokens"))
    output_tokens = _usage_int(usage.get("output_tokens"))
    cache_read_tokens = _usage_int(usage.get("cache_read_input_tokens"))
    cache_creation_tokens = _usage_int(usage.get("cache_creation_input_tokens"))
    total_tokens = sum(
        value
        for value in (input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens)
        if value is not None
    )
    normalized: Dict[str, Any] = {"provider": "claude"}
    if input_tokens is not None:
        normalized["input_tokens"] = input_tokens
    if output_tokens is not None:
        normalized["output_tokens"] = output_tokens
    if cache_read_tokens is not None:
        normalized["cache_read_input_tokens"] = cache_read_tokens
    if cache_creation_tokens is not None:
        normalized["cache_creation_input_tokens"] = cache_creation_tokens
    normalized["total_tokens"] = total_tokens
    return normalized


def _usage_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def parse_decision_output(stdout: str, expected_schema: str) -> AgentDecision:
    payload = stdout.strip()
    if not payload:
        raise ConfigError("Agent hook returned no output.")

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = _extract_embedded_decision_json(payload)
        if parsed is None:
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


def _extract_embedded_decision_json(payload: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    candidates: List[Dict[str, Any]] = []
    for index, char in enumerate(payload):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(payload[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            decision_text = parsed.get("decision")
            if isinstance(decision_text, str) and decision_text.strip():
                candidates.append(parsed)
    if not candidates:
        return None
    return candidates[-1]


def choose_random_legal(
    request: Optional[Dict[str, Any]],
    *,
    rng: Optional[random.Random] = None,
) -> str:
    chooser = rng or random
    if not request:
        return "wait"
    if request.get("wait"):
        return "wait"
    if request.get("teamPreview"):
        side = request.get("side")
        pokemon = side.get("pokemon", []) if isinstance(side, dict) else []
        indexes = [str(index) for index in range(1, len(pokemon) + 1)]
        if not indexes:
            return "wait"
        chooser.shuffle(indexes)
        return f"team {''.join(indexes)}"

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
        chooser.shuffle(available)
        choices: List[str] = []
        available_pool = list(available)
        for needed in force_switch:
            if not needed:
                choices.append("pass")
                continue
            if not available_pool:
                choices.append("pass")
                continue
            target = available_pool.pop()
            choices.append(f"switch {target}")
        return ", ".join(choices)

    active = request.get("active")
    if isinstance(active, list) and active:
        voluntary_switch_choices = _voluntary_switch_choices(request)
        if len(active) == 1 and isinstance(active[0], dict):
            move_choices = _enabled_moves(active[0])
            choices = move_choices + voluntary_switch_choices
            if choices:
                return chooser.choice(choices)
            return "pass"
        choices = []
        for slot in active:
            if not isinstance(slot, dict):
                choices.append("pass")
                continue
            move_choices = _enabled_moves(slot)
            if not move_choices:
                choices.append("pass")
                continue
            choices.append(chooser.choice(move_choices))
        return ", ".join(choices)
    return "wait"


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
            move_choices = _enabled_moves(slot)
            if not move_choices:
                choices.append("pass")
                continue
            choices.append(move_choices[0])
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
        hints.extend(_voluntary_switch_choices(request))

    if not hints:
        hints.append("wait")
    return hints


def _first_enabled_move(active_request: Dict[str, Any]) -> Optional[str]:
    moves = _enabled_moves(active_request)
    return moves[0] if moves else None


def _enabled_moves(active_request: Dict[str, Any]) -> List[str]:
    moves = active_request.get("moves", [])
    if not isinstance(moves, list):
        return []
    enabled: List[str] = []
    for index, move in enumerate(moves, start=1):
        if isinstance(move, dict) and not move.get("disabled", False):
            enabled.append(f"move {index}")
    return enabled


def _can_switch_to(pokemon: Dict[str, Any]) -> bool:
    if pokemon.get("active"):
        return False
    condition = str(pokemon.get("condition") or "")
    return "fnt" not in condition


def _voluntary_switch_choices(request: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(request, dict):
        return []
    if request.get("wait") or request.get("teamPreview"):
        return []
    force_switch = request.get("forceSwitch")
    if isinstance(force_switch, list) and any(bool(slot) for slot in force_switch):
        return []
    active = request.get("active")
    if not isinstance(active, list) or len(active) != 1 or not isinstance(active[0], dict):
        return []
    if active[0].get("trapped") is True:
        return []
    side = request.get("side")
    pokemon = side.get("pokemon", []) if isinstance(side, dict) else []
    choices: List[str] = []
    for index, slot in enumerate(pokemon, start=1):
        if isinstance(slot, dict) and _can_switch_to(slot):
            choices.append(f"switch {index}")
    return choices


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


def _split_protocol_message(payload: str) -> List[tuple[Optional[str], List[str]]]:
    room_id: Optional[str] = None
    current_lines: List[str] = []
    blocks: List[tuple[Optional[str], List[str]]] = []
    for raw_line in payload.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            continue
        if line.startswith(">"):
            if current_lines:
                blocks.append((room_id, current_lines))
            room_id = line[1:] or None
            current_lines = []
            continue
        current_lines.append(line)
    if current_lines:
        blocks.append((room_id, current_lines))
    return blocks


def _user_id(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return "".join(character for character in normalized if character.isalnum())


def _hook_base_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    for key, value in os.environ.items():
        if key in HOOK_ENV_ALLOWLIST or any(key.startswith(prefix) for prefix in HOOK_ENV_PREFIX_ALLOWLIST):
            env[key] = value
    return env
