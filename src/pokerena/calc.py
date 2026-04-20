from __future__ import annotations

import atexit
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import resources
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from jsonschema import Draft202012Validator

from .config import ConfigError
from .transcript import append_transcript_trace_event


CALC_SCRIPT_PATH = Path("tools") / "damage-calc-cli.cjs"
CALC_WORKER_SCRIPT_PATH = Path("tools") / "damage-calc-worker.cjs"
CALC_DEPENDENCY_PATH = Path("node_modules") / "@smogon" / "calc"
CALC_REQUEST_SCHEMA_NAME = "damage-request.v1.json"
CALC_RESULT_SCHEMA_NAME = "damage-result.v1.json"
CALC_REQUEST_SCHEMA_VERSION = "pokerena.damage-request.v1"
CALC_RESULT_SCHEMA_VERSION = "pokerena.damage-result.v1"
CALC_BATCH_REQUEST_SCHEMA_VERSION = "pokerena.damage-batch-request.v1"
CALC_BATCH_RESULT_SCHEMA_VERSION = "pokerena.damage-batch-result.v1"
CALC_SUPPORT_RESULT_SCHEMA_VERSION = "pokerena.damage-support.v1"
CALC_SUPPORT_CACHE_SCHEMA_VERSION = "pokerena.damage-support-cache.v1"
CALC_WORKER_PROTOCOL_VERSION = "pokerena.calc-worker.v1"
CALC_SUPPORT_SUPPORTED_DAMAGING = "supported_damaging"
CALC_SUPPORT_SUPPORTED_NON_DAMAGING = "supported_non_damaging"
CALC_SUPPORT_UNSUPPORTED = "unsupported"
DEFAULT_CALC_TIMEOUT_SECONDS = 15
DEFAULT_CALC_WORKER_STARTUP_SECONDS = 5.0
MAX_CALC_WORKER_LOG_BYTES = 1_000_000
_CALC_WORKER_PROCESSES: Dict[Path, subprocess.Popen[str]] = {}

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


def read_damage_calc_input(
    *,
    input_path: Optional[str],
    use_stdin: bool,
    stdin_text: Optional[str] = None,
) -> Dict[str, Any]:
    payload = _read_json_input(
        input_path=input_path,
        use_stdin=use_stdin,
        stdin_text=stdin_text,
        label="damage calc input",
    )
    if not isinstance(payload, dict):
        raise ConfigError("Damage calc input must be a JSON object.")
    _validate_damage_calc_request(payload)
    return payload


def read_damage_calc_batch_input(
    *,
    input_path: Optional[str],
    use_stdin: bool,
    stdin_text: Optional[str] = None,
) -> Dict[str, Any]:
    payload = _read_json_input(
        input_path=input_path,
        use_stdin=use_stdin,
        stdin_text=stdin_text,
        label="damage calc batch input",
    )
    if not isinstance(payload, dict):
        raise ConfigError("Damage calc batch input must be a JSON object.")
    if payload.get("schema_version") != CALC_BATCH_REQUEST_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CALC_BATCH_REQUEST_SCHEMA_VERSION!r}.")
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ConfigError("requests must be a non-empty list.")
    for index, request_payload in enumerate(requests):
        if not isinstance(request_payload, dict):
            raise ConfigError(f"requests[{index}] must be a JSON object.")
        _validate_damage_calc_request(request_payload)
    return payload


def classify_damage_calc_request(
    payload: Dict[str, Any],
    *,
    project_root: Path,
    timeout_seconds: int = DEFAULT_CALC_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    _validate_damage_calc_request(payload)
    generation = int(payload["generation"])
    move_name = str(payload["move"]["name"]).strip()
    return classify_move_support(
        project_root=project_root,
        generation=generation,
        move_name=move_name,
        timeout_seconds=timeout_seconds,
    )


def run_damage_calc(
    payload: Dict[str, Any],
    *,
    project_root: Path,
    timeout_seconds: int = DEFAULT_CALC_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    _validate_damage_calc_request(payload)
    payload_summary = summarize_damage_calc_payload(payload)
    support = classify_damage_calc_request(
        payload,
        project_root=project_root,
        timeout_seconds=timeout_seconds,
    )
    if support["classification"] == CALC_SUPPORT_SUPPORTED_NON_DAMAGING:
        _trace_calc_helper(
            f"Skipping damage calc for {payload_summary}: move is non-damaging/status.",
            actor_name="calc-worker",
        )
        raise ConfigError(
            f"Damage calc is not applicable for non-damaging move {support['move_name']} in gen {support['generation']}."
        )
    if support["classification"] == CALC_SUPPORT_UNSUPPORTED:
        _trace_calc_helper(
            f"Skipping damage calc for {payload_summary}: move is unsupported by the local calc tool.",
            actor_name="calc-worker",
        )
        raise ConfigError(
            f"Damage calc is unsupported for move {support['move_name']} in gen {support['generation']}."
        )
    _trace_calc_helper(
        f"Running damage calc: {payload_summary}.",
        actor_name="calc-worker",
    )
    response = _worker_request(
        command="damage",
        payload=payload,
        project_root=project_root,
        timeout_seconds=timeout_seconds,
        summary=payload_summary,
    )
    if not isinstance(response, dict):
        raise ConfigError("Damage calc worker returned a non-object response.")
    _validate_schema(CALC_RESULT_SCHEMA_NAME, response)
    _cache_move_support(
        project_root=project_root,
        generation=int(payload["generation"]),
        move_name=str(payload["move"]["name"]),
        classification=CALC_SUPPORT_SUPPORTED_DAMAGING,
    )
    _trace_calc_helper(
        f"Damage calc completed: {payload_summary}.",
        actor_name="calc-worker",
    )
    return response


def run_damage_calc_batch(
    payload: Dict[str, Any],
    *,
    project_root: Path,
    timeout_seconds: int = DEFAULT_CALC_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    if payload.get("schema_version") != CALC_BATCH_REQUEST_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CALC_BATCH_REQUEST_SCHEMA_VERSION!r}.")
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ConfigError("requests must be a non-empty list.")
    for index, request_payload in enumerate(requests):
        if not isinstance(request_payload, dict):
            raise ConfigError(f"requests[{index}] must be a JSON object.")
        _validate_damage_calc_request(request_payload)
    _trace_calc_helper(
        f"Running damage batch for {len(requests)} move candidate(s).",
        actor_name="calc-worker",
    )
    response = _worker_request(
        command="damage-batch",
        payload=payload,
        project_root=project_root,
        timeout_seconds=timeout_seconds,
        summary=f"{len(requests)} calc request(s)",
    )
    results = _validate_damage_calc_batch_result(requests=requests, response=response)
    _cache_batch_support_results(project_root=project_root, requests=requests, results=results)
    skipped_results = 0
    for result in results:
        if result["status"] != "skipped":
            continue
        skipped_results += 1
        reason = "non-damaging/status move" if result["skip_reason"] == "non_damaging" else "unsupported move"
        _trace_calc_helper(
            f"Skipping calc for gen {result['generation']} {result['move_name']}: {reason}.",
            actor_name="calc-worker",
        )
    _trace_calc_helper(
        f"Damage batch completed with {len(results) - skipped_results} calculated result(s) and {skipped_results} skipped move(s).",
        actor_name="calc-worker",
    )
    return response


def sample_damage_calc_payload() -> Dict[str, Any]:
    return {
        "schema_version": CALC_REQUEST_SCHEMA_VERSION,
        "generation": 2,
        "attacker": {
            "species": "Snorlax",
            "options": {
                "level": 100,
                "item": "Leftovers",
                "nature": "Adamant",
                "evs": {"hp": 252, "atk": 252, "def": 252, "spa": 252, "spd": 252, "spe": 252},
            },
        },
        "defender": {
            "species": "Raikou",
            "options": {
                "level": 100,
                "item": "Leftovers",
                "nature": "Timid",
                "evs": {"hp": 252, "atk": 252, "def": 252, "spa": 252, "spd": 252, "spe": 252},
            },
        },
        "move": {
            "name": "Double-Edge",
        },
        "field": {},
    }


def sample_damage_calc_batch_payload() -> Dict[str, Any]:
    return {
        "schema_version": CALC_BATCH_REQUEST_SCHEMA_VERSION,
        "requests": [
            sample_damage_calc_payload(),
            {
                **sample_damage_calc_payload(),
                "move": {"name": "Earthquake"},
            },
        ],
    }


def classify_move_support(
    *,
    project_root: Path,
    generation: int,
    move_name: str,
    timeout_seconds: int = DEFAULT_CALC_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    _ensure_calc_environment(project_root)
    cached = _cached_move_support(
        project_root=project_root,
        generation=generation,
        move_name=move_name,
    )
    if cached is not None:
        return {**cached, "source": "cache"}

    response = _worker_request(
        command="classify-move",
        payload={
            "schema_version": CALC_REQUEST_SCHEMA_VERSION,
            "generation": generation,
            "move": {"name": move_name},
        },
        project_root=project_root,
        timeout_seconds=timeout_seconds,
        summary=f"support check for gen {generation} {move_name}",
    )
    result = _validate_damage_support_result(response)
    _cache_move_support(
        project_root=project_root,
        generation=generation,
        move_name=move_name,
        classification=result["classification"],
    )
    return {**result, "source": "preflight"}


def detect_project_root(start: Optional[Path] = None) -> Path:
    candidate_starts = [Path(start or Path.cwd()).resolve(), Path(__file__).resolve().parents[2]]
    for origin in candidate_starts:
        for candidate in [origin, *origin.parents]:
            if _looks_like_project_root(candidate):
                return candidate
    raise ConfigError(
        "Could not locate the Pokerena project root. Run this command from inside the repository."
    )


def summarize_damage_calc_payload(payload: Dict[str, Any]) -> str:
    attacker = _payload_name(payload.get("attacker"), "species")
    defender = _payload_name(payload.get("defender"), "species")
    move = _payload_name(payload.get("move"), "name")
    generation = payload.get("generation", "?")
    return f"gen {generation} {attacker} using {move} into {defender}"


def default_calc_support_cache_path(project_root: Path) -> Path:
    return project_root / ".runtime" / "calc" / "move-support-cache.json"


def _worker_request(
    *,
    command: str,
    payload: Dict[str, Any],
    project_root: Path,
    timeout_seconds: int,
    summary: str,
) -> Dict[str, Any]:
    _ensure_calc_environment(project_root)
    socket_path = _ensure_calc_worker(project_root, timeout_seconds=DEFAULT_CALC_WORKER_STARTUP_SECONDS)
    request = {"command": command, "payload": payload}
    try:
        response = _send_worker_request(socket_path, request, timeout_seconds=timeout_seconds)
    except TimeoutError as error:
        raise ConfigError(
            f"Damage calc command timed out after {timeout_seconds} seconds for {summary}."
        ) from error
    if not isinstance(response, dict):
        raise ConfigError("Damage calc worker returned an invalid response.")
    if not response.get("ok", False):
        detail = str(response.get("error") or "unknown error")
        raise ConfigError(f"Damage calc command failed for {summary}: {detail}")
    result = response.get("result")
    if not isinstance(result, dict):
        raise ConfigError("Damage calc worker returned an invalid result payload.")
    return result


def _validate_damage_support_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if result.get("schema_version") != CALC_SUPPORT_RESULT_SCHEMA_VERSION:
        raise ConfigError(f"Damage support result must declare {CALC_SUPPORT_RESULT_SCHEMA_VERSION!r}.")
    generation = result.get("generation")
    move_name = result.get("move_name")
    classification = result.get("classification")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise ConfigError("Damage support result must include a positive integer generation.")
    if not isinstance(move_name, str) or not move_name.strip():
        raise ConfigError("Damage support result must include move_name.")
    if classification not in {
        CALC_SUPPORT_SUPPORTED_DAMAGING,
        CALC_SUPPORT_SUPPORTED_NON_DAMAGING,
        CALC_SUPPORT_UNSUPPORTED,
    }:
        raise ConfigError("Damage support result has an invalid classification.")
    return {
        "schema_version": CALC_SUPPORT_RESULT_SCHEMA_VERSION,
        "generation": generation,
        "move_name": move_name.strip(),
        "classification": classification,
        "reason": str(result.get("reason") or ""),
    }


def _validate_damage_calc_batch_result(
    *,
    requests: List[Dict[str, Any]],
    response: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not isinstance(response, dict):
        raise ConfigError("Damage calc worker returned a non-object response.")
    if response.get("schema_version") != CALC_BATCH_RESULT_SCHEMA_VERSION:
        raise ConfigError(f"Damage batch result must declare {CALC_BATCH_RESULT_SCHEMA_VERSION!r}.")
    results = response.get("results")
    if not isinstance(results, list) or len(results) != len(requests):
        raise ConfigError("Damage batch result must include one result per request.")

    validated: List[Dict[str, Any]] = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ConfigError(f"results[{index}] must be a JSON object.")
        status = result.get("status")
        move_name = result.get("move_name")
        generation = result.get("generation")
        if not isinstance(move_name, str) or not move_name.strip():
            raise ConfigError(f"results[{index}].move_name must be a non-empty string.")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ConfigError(f"results[{index}].generation must be a positive integer.")
        if status == "ok":
            payload = result.get("result")
            if not isinstance(payload, dict):
                raise ConfigError(f"results[{index}].result must be a JSON object when status is 'ok'.")
            _validate_schema(CALC_RESULT_SCHEMA_NAME, payload)
            validated.append(
                {
                    "status": "ok",
                    "move_name": move_name.strip(),
                    "generation": generation,
                    "result": payload,
                }
            )
            continue
        if status == "skipped":
            skip_reason = result.get("skip_reason")
            if skip_reason not in {"non_damaging", "unsupported"}:
                raise ConfigError(
                    f"results[{index}].skip_reason must be 'non_damaging' or 'unsupported' when status is 'skipped'."
                )
            validated.append(
                {
                    "status": "skipped",
                    "skip_reason": skip_reason,
                    "move_name": move_name.strip(),
                    "generation": generation,
                }
            )
            continue
        raise ConfigError(f"results[{index}].status must be 'ok' or 'skipped'.")
    return validated


def _cache_batch_support_results(
    *,
    project_root: Path,
    requests: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
) -> None:
    for request_payload, result in zip(requests, results):
        move_name = _payload_name(request_payload.get("move"), "name")
        generation = request_payload.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int):
            continue
        classification = CALC_SUPPORT_SUPPORTED_DAMAGING
        if result["status"] == "skipped":
            classification = (
                CALC_SUPPORT_SUPPORTED_NON_DAMAGING
                if result["skip_reason"] == "non_damaging"
                else CALC_SUPPORT_UNSUPPORTED
            )
        _cache_move_support(
            project_root=project_root,
            generation=generation,
            move_name=move_name,
            classification=classification,
        )


def _cached_move_support(
    *,
    project_root: Path,
    generation: int,
    move_name: str,
) -> Optional[Dict[str, Any]]:
    payload = _read_calc_support_cache(project_root)
    cache_entries = payload.get("moves")
    if not isinstance(cache_entries, dict):
        return None
    entry = cache_entries.get(_move_support_key(generation, move_name))
    if not isinstance(entry, dict):
        return None
    classification = entry.get("classification")
    if classification not in {
        CALC_SUPPORT_SUPPORTED_DAMAGING,
        CALC_SUPPORT_SUPPORTED_NON_DAMAGING,
        CALC_SUPPORT_UNSUPPORTED,
    }:
        return None
    return {
        "schema_version": CALC_SUPPORT_RESULT_SCHEMA_VERSION,
        "generation": generation,
        "move_name": str(entry.get("move_name") or move_name).strip(),
        "classification": classification,
        "reason": str(entry.get("reason") or "cache"),
    }


def _cache_move_support(
    *,
    project_root: Path,
    generation: int,
    move_name: str,
    classification: str,
    reason: Optional[str] = None,
) -> None:
    if classification not in {
        CALC_SUPPORT_SUPPORTED_DAMAGING,
        CALC_SUPPORT_SUPPORTED_NON_DAMAGING,
        CALC_SUPPORT_UNSUPPORTED,
    }:
        return
    payload = _read_calc_support_cache(project_root)
    moves = payload.setdefault("moves", {})
    if not isinstance(moves, dict):
        moves = {}
        payload["moves"] = moves
    moves[_move_support_key(generation, move_name)] = {
        "generation": generation,
        "move_name": str(move_name).strip(),
        "classification": classification,
        "reason": reason or classification,
        "updated_at": _timestamp(),
    }
    _write_calc_support_cache(project_root, payload)


def _read_calc_support_cache(project_root: Path) -> Dict[str, Any]:
    cache_path = default_calc_support_cache_path(project_root)
    calc_version = _calc_dependency_version(project_root)
    if not cache_path.exists():
        return _empty_calc_support_cache(calc_version)
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_calc_support_cache(calc_version)
    if not isinstance(payload, dict):
        return _empty_calc_support_cache(calc_version)
    if payload.get("schema_version") != CALC_SUPPORT_CACHE_SCHEMA_VERSION:
        return _empty_calc_support_cache(calc_version)
    if payload.get("calc_version") != calc_version:
        return _empty_calc_support_cache(calc_version)
    moves = payload.get("moves")
    if not isinstance(moves, dict):
        return _empty_calc_support_cache(calc_version)
    return payload


def _write_calc_support_cache(project_root: Path, payload: Dict[str, Any]) -> None:
    cache_path = default_calc_support_cache_path(project_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _empty_calc_support_cache(calc_version: str) -> Dict[str, Any]:
    return {
        "schema_version": CALC_SUPPORT_CACHE_SCHEMA_VERSION,
        "calc_version": calc_version,
        "moves": {},
    }


def _move_support_key(generation: int, move_name: str) -> str:
    normalized_name = " ".join(str(move_name).strip().lower().split())
    return f"gen{generation}:{normalized_name}"


def _calc_dependency_version(project_root: Path) -> str:
    package_path = project_root / "node_modules" / "@smogon" / "calc" / "package.json"
    if not package_path.exists():
        return "unknown"
    try:
        payload = json.loads(package_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    version = payload.get("version")
    if isinstance(version, str) and version.strip():
        return version.strip()
    return "unknown"


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_calc_environment(project_root: Path) -> None:
    if shutil.which("node") is None:
        raise ConfigError("node is required on PATH. Run ./scripts/bootstrap-node-deps.sh first.")
    script_path = project_root / CALC_SCRIPT_PATH
    worker_script_path = project_root / CALC_WORKER_SCRIPT_PATH
    dependency_path = project_root / CALC_DEPENDENCY_PATH
    if not script_path.exists():
        raise ConfigError(f"Damage calc script not found at {script_path}.")
    if not worker_script_path.exists():
        raise ConfigError(f"Damage calc worker script not found at {worker_script_path}.")
    if not dependency_path.exists():
        raise ConfigError(
            f"{dependency_path} is missing. Run ./scripts/bootstrap-node-deps.sh first."
        )
    if not hasattr(socket, "AF_UNIX"):
        raise ConfigError("The local calc worker requires UNIX socket support.")


def _ensure_calc_worker(project_root: Path, *, timeout_seconds: float) -> Path:
    runtime_dir = project_root / ".runtime" / "calc"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    socket_path = runtime_dir / "worker.sock"
    log_path = runtime_dir / "worker.log"
    lock_path = runtime_dir / "worker.lock"

    with _calc_worker_start_lock(lock_path):
        existing_process = _CALC_WORKER_PROCESSES.get(socket_path)
        if existing_process is not None:
            if existing_process.poll() is None and _ping_worker(socket_path):
                return socket_path
            _CALC_WORKER_PROCESSES.pop(socket_path, None)

        if _ping_worker(socket_path):
            return socket_path
        if socket_path.exists():
            socket_path.unlink()

        _rotate_worker_log_if_needed(log_path)
        with log_path.open("a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                ["node", str(project_root / CALC_WORKER_SCRIPT_PATH), "--socket", str(socket_path)],
                cwd=project_root,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
            _CALC_WORKER_PROCESSES[socket_path] = process

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                _CALC_WORKER_PROCESSES.pop(socket_path, None)
                raise ConfigError(
                    f"Damage calc worker exited before startup completed (code {process.returncode})."
                )
            if _ping_worker(socket_path):
                return socket_path
            time.sleep(0.05)
        _CALC_WORKER_PROCESSES.pop(socket_path, None)
        raise ConfigError(f"Timed out waiting for the damage calc worker at {socket_path}.")


def _ping_worker(socket_path: Path) -> bool:
    if not socket_path.exists():
        return False
    try:
        response = _send_worker_request(
            socket_path,
            {"command": "ping"},
            timeout_seconds=1,
        )
    except (OSError, TimeoutError, ValueError):
        return False
    if not isinstance(response, dict) or not bool(response.get("ok")):
        return False
    result = response.get("result")
    if not isinstance(result, dict):
        return False
    if result.get("protocol_version") != CALC_WORKER_PROTOCOL_VERSION:
        return False
    commands = result.get("commands")
    if not isinstance(commands, list):
        return False
    command_set = {command for command in commands if isinstance(command, str)}
    required_commands = {"damage", "damage-batch", "classify-move"}
    return required_commands.issubset(command_set)


def _send_worker_request(
    socket_path: Path,
    payload: Dict[str, Any],
    *,
    timeout_seconds: int | float,
) -> Dict[str, Any]:
    if not socket_path.exists():
        raise OSError(f"Worker socket does not exist: {socket_path}")
    deadline = time.monotonic() + float(timeout_seconds)
    message = json.dumps(payload) + "\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(_remaining_timeout(deadline))
        client.connect(str(socket_path))
        client.settimeout(_remaining_timeout(deadline))
        client.sendall(message.encode("utf-8"))
        client.shutdown(socket.SHUT_WR)
        chunks: List[bytes] = []
        while True:
            try:
                client.settimeout(_remaining_timeout(deadline))
                chunk = client.recv(65536)
            except socket.timeout as error:
                raise TimeoutError("Timed out waiting for calc worker response.") from error
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    raw = b"".join(chunks).decode("utf-8").strip()
    if not raw:
        raise ValueError("Damage calc worker returned no output.")
    return json.loads(raw)


@contextmanager
def _calc_worker_start_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _rotate_worker_log_if_needed(log_path: Path) -> None:
    try:
        if log_path.exists() and log_path.stat().st_size >= MAX_CALC_WORKER_LOG_BYTES:
            rotated_path = log_path.with_suffix(".log.1")
            if rotated_path.exists():
                rotated_path.unlink()
            log_path.replace(rotated_path)
    except OSError:
        return


def _remaining_timeout(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Timed out waiting for calc worker response.")
    return remaining


def _read_json_input(
    *,
    input_path: Optional[str],
    use_stdin: bool,
    stdin_text: Optional[str],
    label: str,
) -> Any:
    if bool(input_path) == bool(use_stdin):
        raise ConfigError("Choose exactly one of --input or --stdin.")

    if use_stdin:
        raw_text = stdin_text if stdin_text is not None else sys.stdin.read()
        source = "stdin"
    else:
        path = Path(input_path).resolve() if input_path is not None else None
        if path is None or not path.exists():
            raise ConfigError(f"{label.capitalize()} file not found at {input_path}.")
        raw_text = path.read_text(encoding="utf-8")
        source = str(path)

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ConfigError(f"Failed to parse JSON from {source}: {error}") from error


def _validate_damage_calc_request(payload: Dict[str, Any]) -> None:
    _require_schema_version(payload.get("schema_version"), CALC_REQUEST_SCHEMA_VERSION)
    _require_positive_int(payload.get("generation"), "generation")
    _require_named_object(payload.get("attacker"), "attacker", "species")
    _require_named_object(payload.get("defender"), "defender", "species")
    _require_named_object(payload.get("move"), "move", "name")
    _require_optional_object(payload.get("field"), "field")
    _validate_schema(CALC_REQUEST_SCHEMA_NAME, payload)


def _trace_calc_helper(message: str, *, actor_name: str) -> None:
    transcript_path = os.environ.get("POKERENA_TRANSCRIPT_PATH")
    request_sequence = os.environ.get("POKERENA_REQUEST_SEQUENCE")
    decision_attempt = os.environ.get("POKERENA_DECISION_ATTEMPT")
    if not transcript_path or not request_sequence:
        return
    try:
        append_transcript_trace_event(
            Path(transcript_path),
            request_sequence=int(request_sequence),
            decision_attempt=int(decision_attempt) if decision_attempt else None,
            kind="status",
            message=message,
            actor_kind="helper",
            actor_name=actor_name,
        )
    except Exception:
        return


def _require_positive_int(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{label} must be a positive integer.")


def _require_named_object(value: Any, label: str, key: str) -> None:
    mapping = _require_optional_object(value, label, allow_missing=False)
    field = mapping.get(key)
    if not isinstance(field, str) or not field.strip():
        raise ConfigError(f"{label}.{key} must be a non-empty string.")
    _require_optional_object(mapping.get("options"), f"{label}.options")


def _require_optional_object(
    value: Any,
    label: str,
    *,
    allow_missing: bool = True,
) -> Dict[str, Any]:
    if value is None and allow_missing:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a JSON object.")
    return value


def _require_schema_version(value: Any, expected: str) -> None:
    if value != expected:
        raise ConfigError(f"schema_version must be {expected!r}.")


def _payload_name(value: Any, key: str) -> str:
    if isinstance(value, dict):
        field = value.get(key)
        if isinstance(field, str) and field.strip():
            return field
    return "unknown"


def _looks_like_project_root(path: Path) -> bool:
    return (
        (path / CALC_SCRIPT_PATH).exists()
        and (path / "package.json").exists()
        and (path / "vendor" / "pokemon-showdown").exists()
    )


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


def _cleanup_calc_workers() -> None:
    for socket_path, process in list(_CALC_WORKER_PROCESSES.items()):
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=2)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=2)
            except Exception:
                pass
        finally:
            _CALC_WORKER_PROCESSES.pop(socket_path, None)


atexit.register(_cleanup_calc_workers)
