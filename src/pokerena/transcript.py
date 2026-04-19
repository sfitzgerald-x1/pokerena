from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from send2trash import send2trash

from .config import AgentDefinition
from .pricing import estimate_usage_cost, pricing_snapshot_for_agent, resolve_agent_model


TRANSCRIPT_SCHEMA_VERSION = "pokerena.agent-transcript.v1"


class BattleSessionDeleteConflictError(ValueError):
    """Raised when the caller tries to delete a live battle session."""


class BattleSessionStopConflictError(ValueError):
    """Raised when the caller tries to stop a finished battle session."""


@dataclass(frozen=True)
class TranscriptTraceEvent:
    kind: str
    message: str
    created_at: str
    actor_kind: str = "pokerena"
    actor_name: Optional[str] = None


@dataclass(frozen=True)
class TranscriptEntry:
    turn_number: Optional[int]
    request_sequence: int
    request_kind: str
    rqid: Optional[str]
    decision_attempt: int
    prompt_text: str
    recent_public_events: List[str]
    decision: Optional[str] = None
    raw_output: str = ""
    notes: str = ""
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    error: Optional[str] = None
    entry_kind: str = "turn"
    turn_state: str = "thinking"
    submission_state: str = "pending"
    submission_detail: Optional[str] = None
    selected_action: Optional[str] = None
    selected_action_label: Optional[str] = None
    selected_action_source: Optional[str] = None
    timeout_seconds: Optional[int] = None
    trace_events: List[TranscriptTraceEvent] = field(default_factory=list)
    decision_latency_ms: Optional[int] = None
    usage: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    submitted_at: Optional[str] = None
    timed_out_at: Optional[str] = None
    validated_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: _timestamp())


def default_transcript_path(runtime_root: Path, agent: AgentDefinition, battle_id: str) -> Path:
    return runtime_root / "agents" / agent.agent_id / battle_id / "transcript.json"


def default_battle_control_path(runtime_root: Path, agent_id: str, battle_id: str) -> Path:
    return runtime_root / "agents" / agent_id / battle_id / "control.json"


def record_transcript_entry(
    path: Path,
    *,
    agent: AgentDefinition,
    battle_id: str,
    format_name: Optional[str],
    challenger: Optional[str],
    winner: Optional[str],
    finished: bool,
    entry: TranscriptEntry,
) -> None:
    payload = _load_or_create_payload(path, agent=agent, battle_id=battle_id)
    _merge_agent_metadata(payload, agent)
    _merge_metadata(
        payload,
        format_name=format_name,
        challenger=challenger,
        winner=winner,
        finished=finished,
    )
    payload["entries"].append(asdict(entry))
    payload["updated_at"] = _timestamp()
    _write_payload(path, payload)


def update_transcript_metadata(
    path: Path,
    *,
    agent: AgentDefinition,
    battle_id: str,
    format_name: Optional[str] = None,
    challenger: Optional[str] = None,
    winner: Optional[str] = None,
    finished: Optional[bool] = None,
) -> None:
    payload = _load_or_create_payload(path, agent=agent, battle_id=battle_id)
    _merge_agent_metadata(payload, agent)
    _merge_metadata(
        payload,
        format_name=format_name,
        challenger=challenger,
        winner=winner,
        finished=finished if finished is not None else bool(payload.get("finished")),
    )
    payload["updated_at"] = _timestamp()
    _write_payload(path, payload)


def update_transcript_entry_state(
    path: Path,
    *,
    request_sequence: int,
    decision_attempt: Optional[int] = None,
    submission_state: str,
    submission_detail: Optional[str] = None,
    turn_state: Optional[str] = None,
) -> None:
    validated_at = _timestamp()
    if turn_state is None and submission_state in {"accepted", "rejected"}:
        turn_state = submission_state
    update_transcript_entry(
        path,
        request_sequence=request_sequence,
        decision_attempt=decision_attempt,
        submission_state=submission_state,
        submission_detail=submission_detail,
        validated_at=validated_at,
        turn_state=turn_state,
    )


def update_transcript_entry(
    path: Path,
    *,
    request_sequence: int,
    decision_attempt: Optional[int] = None,
    **changes: Any,
) -> None:
    payload = _read_payload(path)
    entry = _find_entry(payload, request_sequence=request_sequence, decision_attempt=decision_attempt)
    if entry is None:
        raise ValueError(
            f"Transcript at {path} has no entry for request_sequence={request_sequence}"
            + (f" decision_attempt={decision_attempt}" if decision_attempt is not None else "")
            + "."
        )
    for key, value in changes.items():
        entry[key] = value
    payload["updated_at"] = _timestamp()
    _write_payload(path, payload)


def append_transcript_trace_event(
    path: Path,
    *,
    request_sequence: int,
    kind: str,
    message: str,
    decision_attempt: Optional[int] = None,
    actor_kind: Optional[str] = None,
    actor_name: Optional[str] = None,
) -> None:
    if not message:
        return
    payload = _read_payload(path)
    entry = _find_entry(payload, request_sequence=request_sequence, decision_attempt=decision_attempt)
    if entry is None:
        raise ValueError(
            f"Transcript at {path} has no entry for request_sequence={request_sequence}"
            + (f" decision_attempt={decision_attempt}" if decision_attempt is not None else "")
            + "."
        )
    trace_events = entry.get("trace_events", [])
    if not isinstance(trace_events, list):
        raise ValueError(f"Transcript at {path} has invalid trace_events data.")
    resolved_actor_kind = actor_kind or _default_actor_kind(kind)
    event_payload = asdict(
        TranscriptTraceEvent(
            kind=kind,
            message=message,
            created_at=_timestamp(),
            actor_kind=resolved_actor_kind,
            actor_name=actor_name,
        )
    )
    if (
        trace_events
        and isinstance(trace_events[-1], dict)
        and trace_events[-1].get("kind") == kind
        and trace_events[-1].get("actor_kind") == resolved_actor_kind
        and trace_events[-1].get("actor_name") == actor_name
        and resolved_actor_kind == "agent"
    ):
        trace_events[-1]["message"] = str(trace_events[-1].get("message") or "") + message
    else:
        trace_events.append(event_payload)
    entry["trace_events"] = trace_events
    payload["updated_at"] = _timestamp()
    _write_payload(path, payload)


def list_transcript_summaries(runtime_root: Path) -> List[Dict[str, Any]]:
    transcripts_root = runtime_root / "agents"
    if not transcripts_root.exists():
        return []

    summaries: List[Dict[str, Any]] = []
    for path in transcripts_root.glob("*/*/transcript.json"):
        payload = _read_payload(path)
        entries = payload.get("entries", [])
        helper_summary = summarize_helper_activity(entries)
        control_payload = load_battle_stop_request(runtime_root, str(payload.get("agent_id") or ""), str(payload.get("battle_id") or ""))
        summaries.append(
            {
                "battle_id": payload.get("battle_id"),
                "agent_id": payload.get("agent_id"),
                "provider": payload.get("provider"),
                "transport": payload.get("transport"),
                "format_name": payload.get("format_name"),
                "challenger": payload.get("challenger"),
                "winner": payload.get("winner"),
                "finished": bool(payload.get("finished")),
                "finished_at": payload.get("finished_at"),
                "started_at": payload.get("started_at"),
                "updated_at": payload.get("updated_at"),
                "stop_requested_at": control_payload.get("requested_at") if control_payload else None,
                "stop_handled_at": control_payload.get("handled_at") if control_payload else None,
                "entry_count": len(entries) if isinstance(entries, list) else 0,
                "helper_summary": helper_summary,
            }
        )
    summaries.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return summaries


def load_battle_transcript(runtime_root: Path, agent_id: str, battle_id: str) -> Optional[Dict[str, Any]]:
    path = runtime_root / "agents" / agent_id / battle_id / "transcript.json"
    if not path.exists():
        return None
    payload = _read_payload(path)
    control_payload = load_battle_stop_request(runtime_root, agent_id, battle_id)
    payload["helper_summary"] = summarize_helper_activity(payload.get("entries", []))
    payload["stop_requested_at"] = control_payload.get("requested_at") if control_payload else None
    payload["stop_handled_at"] = control_payload.get("handled_at") if control_payload else None
    return payload


def load_transcript_entry(
    path: Path,
    *,
    request_sequence: int,
    decision_attempt: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    payload = _read_payload(path)
    return _find_entry(payload, request_sequence=request_sequence, decision_attempt=decision_attempt)


def upsert_battle_summary_entry(path: Path) -> Dict[str, Any]:
    payload = _read_payload(path)
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"Transcript at {path} has invalid entries data.")

    turn_entries = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("entry_kind", "turn") == "turn"
    ]
    helper_summary = summarize_helper_activity(turn_entries)
    usage = aggregate_usage(turn_entries)
    total_turns = max(
        (int(entry.get("turn_number")) for entry in turn_entries if isinstance(entry.get("turn_number"), int)),
        default=0,
    )
    latencies = [
        int(entry["decision_latency_ms"])
        for entry in turn_entries
        if isinstance(entry, dict) and isinstance(entry.get("decision_latency_ms"), int)
    ]
    average_latency_ms = round(sum(latencies) / len(latencies)) if latencies else None
    result_label = payload.get("winner") or "tie"
    request_sequence = max(
        (
            int(entry.get("request_sequence", 0))
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("request_sequence"), int)
        ),
        default=0,
    ) + 1

    summary_payload: Dict[str, Any] = {
        "result": result_label,
        "total_turns": total_turns,
        "average_decision_latency_ms": average_latency_ms,
        "helper_summary": helper_summary,
    }
    if usage:
        summary_payload["usage"] = usage
    cost_estimate = estimate_usage_cost(
        usage,
        model=payload.get("model") if isinstance(payload.get("model"), str) else None,
        pricing_snapshot=payload.get("pricing") if isinstance(payload.get("pricing"), dict) else None,
    )
    if cost_estimate:
        summary_payload["cost_estimate"] = cost_estimate

    summary_entry = asdict(
        TranscriptEntry(
            entry_kind="summary",
            turn_number=total_turns or None,
            request_sequence=request_sequence,
            request_kind="summary",
            rqid=None,
            decision_attempt=1,
            prompt_text="",
            recent_public_events=[],
            decision=None,
            raw_output="",
            notes="",
            fallback_used=False,
            fallback_reason=None,
            error=None,
            turn_state="accepted",
            submission_state="accepted",
            submission_detail="Battle finished and summary recorded.",
            selected_action=None,
            selected_action_label=None,
            selected_action_source=None,
            timeout_seconds=None,
            trace_events=[],
            decision_latency_ms=None,
            usage=usage,
            summary=summary_payload,
            submitted_at=None,
            timed_out_at=None,
            validated_at=_timestamp(),
            created_at=_timestamp(),
        )
    )

    replaced = False
    for index, entry in enumerate(entries):
        if isinstance(entry, dict) and entry.get("entry_kind") == "summary":
            entries[index] = summary_entry
            replaced = True
            break
    if not replaced:
        entries.append(summary_entry)
    payload["entries"] = entries
    payload["updated_at"] = _timestamp()
    _write_payload(path, payload)
    return summary_entry


def delete_battle_session(runtime_root: Path, agent_id: str, battle_id: str) -> None:
    session_dir = (runtime_root / "agents" / agent_id / battle_id).resolve()
    agents_root = (runtime_root / "agents").resolve()
    if not _is_relative_to(session_dir, agents_root):
        raise ValueError("Battle session path escaped the runtime directory.")
    transcript_path = session_dir / "transcript.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Battle session {agent_id}/{battle_id} was not found.")
    payload = _read_payload(transcript_path)
    if not bool(payload.get("finished")):
        raise BattleSessionDeleteConflictError("Live battle sessions cannot be deleted.")
    send2trash(str(session_dir))


def request_battle_stop(runtime_root: Path, agent_id: str, battle_id: str) -> Dict[str, Any]:
    session_dir = (runtime_root / "agents" / agent_id / battle_id).resolve()
    agents_root = (runtime_root / "agents").resolve()
    if not _is_relative_to(session_dir, agents_root):
        raise ValueError("Battle session path escaped the runtime directory.")
    transcript_path = session_dir / "transcript.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Battle session {agent_id}/{battle_id} was not found.")
    payload = _read_payload(transcript_path)
    if bool(payload.get("finished")):
        raise BattleSessionStopConflictError("Finished battle sessions cannot be stopped.")

    control_path = default_battle_control_path(runtime_root, agent_id, battle_id)
    control_payload = _read_control_payload(control_path) or {}
    if not control_payload.get("requested_at"):
        control_payload["action"] = "forfeit"
        control_payload["requested_at"] = _timestamp()
    _write_control_payload(control_path, control_payload)
    return control_payload


def load_battle_stop_request(runtime_root: Path, agent_id: str, battle_id: str) -> Optional[Dict[str, Any]]:
    control_path = default_battle_control_path(runtime_root, agent_id, battle_id)
    return _read_control_payload(control_path)


def mark_battle_stop_handled(
    runtime_root: Path,
    agent_id: str,
    battle_id: str,
) -> Optional[Dict[str, Any]]:
    control_path = default_battle_control_path(runtime_root, agent_id, battle_id)
    control_payload = _read_control_payload(control_path)
    if control_payload is None:
        return None
    if not control_payload.get("handled_at"):
        control_payload["handled_at"] = _timestamp()
        _write_control_payload(control_path, control_payload)
    return control_payload


def clear_battle_stop_request(runtime_root: Path, agent_id: str, battle_id: str) -> None:
    control_path = default_battle_control_path(runtime_root, agent_id, battle_id)
    if control_path.exists():
        control_path.unlink()


def summarize_helper_activity(entries: object) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    total_events = 0
    if not isinstance(entries, list):
        return {"count": 0, "actors": []}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        trace_events = entry.get("trace_events", [])
        if not isinstance(trace_events, list):
            continue
        for event in trace_events:
            if not isinstance(event, dict):
                continue
            if event.get("actor_kind") != "helper":
                continue
            actor_name = str(event.get("actor_name") or "helper")
            counts[actor_name] = counts.get(actor_name, 0) + 1
            total_events += 1
    actors = [
        {"name": name, "count": count}
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {"count": total_events, "actors": actors}


def aggregate_usage(entries: object) -> Optional[Dict[str, Any]]:
    if not isinstance(entries, list):
        return None
    totals: Dict[str, int] = {}
    provider: Optional[str] = None
    seen = False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        usage = entry.get("usage")
        if not isinstance(usage, dict):
            continue
        seen = True
        if provider is None and isinstance(usage.get("provider"), str):
            provider = usage["provider"]
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    if not seen:
        return None
    if provider is not None:
        totals["provider"] = provider
    return totals


def _load_or_create_payload(path: Path, *, agent: AgentDefinition, battle_id: str) -> Dict[str, Any]:
    if path.exists():
        return _read_payload(path)
    timestamp = _timestamp()
    return {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "battle_id": battle_id,
        "agent_id": agent.agent_id,
        "provider": agent.provider,
        "transport": agent.transport,
        "player_slot": agent.player_slot,
        "model": resolve_agent_model(agent),
        "pricing": pricing_snapshot_for_agent(agent),
        "format_name": None,
        "challenger": None,
        "winner": None,
        "finished": False,
        "finished_at": None,
        "started_at": timestamp,
        "updated_at": timestamp,
        "entries": [],
    }


def _find_entry(
    payload: Dict[str, Any],
    *,
    request_sequence: int,
    decision_attempt: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Transcript payload has invalid entries data.")
    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        if entry.get("request_sequence") != request_sequence:
            continue
        if decision_attempt is not None and entry.get("decision_attempt") != decision_attempt:
            continue
        return entry
    return None


def _merge_metadata(
    payload: Dict[str, Any],
    *,
    format_name: Optional[str],
    challenger: Optional[str],
    winner: Optional[str],
    finished: bool,
) -> None:
    if format_name:
        payload["format_name"] = format_name
    if challenger:
        payload["challenger"] = challenger
    if winner is not None:
        payload["winner"] = winner
    effective_finished = bool(payload.get("finished")) or finished or winner is not None
    payload["finished"] = effective_finished
    if effective_finished:
        payload["finished_at"] = payload.get("finished_at") or _timestamp()
    else:
        payload["finished_at"] = None


def _read_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Transcript at {path} must contain a JSON object.")
    payload, changed = _normalize_payload(payload)
    if changed:
        _write_payload(path, payload)
    return payload


def _write_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _default_actor_kind(kind: str) -> str:
    if kind == "agent":
        return "agent"
    return "pokerena"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _normalize_payload(payload: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    changed = False
    if "entries" not in payload or not isinstance(payload.get("entries"), list):
        payload["entries"] = []
        changed = True
    if "transport" not in payload:
        payload["transport"] = None
        changed = True
    if "model" not in payload:
        payload["model"] = None
        changed = True
    if "pricing" not in payload:
        payload["pricing"] = None
        changed = True
    effective_finished = _payload_effective_finished(payload)
    if bool(payload.get("finished")) != effective_finished:
        payload["finished"] = effective_finished
        changed = True
    inferred_finished_at = _infer_finished_at(payload) if effective_finished else None
    if payload.get("finished_at") != inferred_finished_at:
        payload["finished_at"] = inferred_finished_at
        changed = True
    return payload, changed


def _merge_agent_metadata(payload: Dict[str, Any], agent: AgentDefinition) -> None:
    payload["provider"] = agent.provider
    payload["transport"] = agent.transport
    payload["player_slot"] = agent.player_slot
    payload["model"] = resolve_agent_model(agent)
    payload["pricing"] = pricing_snapshot_for_agent(agent)


def _payload_effective_finished(payload: Dict[str, Any]) -> bool:
    if bool(payload.get("finished")):
        return True
    winner = payload.get("winner")
    if isinstance(winner, str) and winner.strip():
        return True
    if payload.get("finished_at"):
        return True
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return False
    return any(
        isinstance(entry, dict) and entry.get("entry_kind") == "summary"
        for entry in entries
    )


def _infer_finished_at(payload: Dict[str, Any]) -> Optional[str]:
    finished_at = payload.get("finished_at")
    if isinstance(finished_at, str) and finished_at:
        return finished_at
    entries = payload.get("entries", [])
    if isinstance(entries, list):
        for entry in reversed(entries):
            if not isinstance(entry, dict):
                continue
            if entry.get("entry_kind") == "summary":
                for key in ("validated_at", "created_at"):
                    value = entry.get(key)
                    if isinstance(value, str) and value:
                        return value
    for key in ("updated_at", "started_at"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _read_control_payload(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Battle control file at {path} must contain a JSON object.")
    return payload


def _write_control_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
