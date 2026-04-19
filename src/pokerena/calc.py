from __future__ import annotations

from importlib import resources
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

from jsonschema import Draft202012Validator

from .config import ConfigError


CALC_SCRIPT_PATH = Path("tools") / "damage-calc-cli.cjs"
CALC_DEPENDENCY_PATH = Path("node_modules") / "@smogon" / "calc"
CALC_REQUEST_SCHEMA_NAME = "damage-request.v1.json"
CALC_RESULT_SCHEMA_NAME = "damage-result.v1.json"
CALC_REQUEST_SCHEMA_VERSION = "pokerena.damage-request.v1"
CALC_RESULT_SCHEMA_VERSION = "pokerena.damage-result.v1"
DEFAULT_CALC_TIMEOUT_SECONDS = 15


def read_damage_calc_input(
    *,
    input_path: Optional[str],
    use_stdin: bool,
    stdin_text: Optional[str] = None,
) -> Dict[str, Any]:
    if bool(input_path) == bool(use_stdin):
        raise ConfigError("Choose exactly one of --input or --stdin.")

    if use_stdin:
        raw_text = stdin_text if stdin_text is not None else sys.stdin.read()
        source = "stdin"
    else:
        path = Path(input_path).resolve() if input_path is not None else None
        if path is None or not path.exists():
            raise ConfigError(f"Damage calc input file not found at {input_path}.")
        raw_text = path.read_text(encoding="utf-8")
        source = str(path)

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ConfigError(f"Failed to parse damage calc JSON from {source}: {error}") from error

    if not isinstance(payload, dict):
        raise ConfigError("Damage calc input must be a JSON object.")

    _require_schema_version(payload.get("schema_version"), CALC_REQUEST_SCHEMA_VERSION)
    _require_positive_int(payload.get("generation"), "generation")
    _require_named_object(payload.get("attacker"), "attacker", "species")
    _require_named_object(payload.get("defender"), "defender", "species")
    _require_named_object(payload.get("move"), "move", "name")
    _require_optional_object(payload.get("field"), "field")
    _validate_schema(CALC_REQUEST_SCHEMA_NAME, payload)

    return payload


def run_damage_calc(
    payload: Dict[str, Any],
    *,
    project_root: Path,
    timeout_seconds: int = DEFAULT_CALC_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    script_path = project_root / CALC_SCRIPT_PATH
    dependency_path = project_root / CALC_DEPENDENCY_PATH
    payload_summary = summarize_damage_calc_payload(payload)

    if shutil.which("node") is None:
        raise ConfigError("node is required on PATH. Run ./scripts/bootstrap-node-deps.sh first.")
    if not script_path.exists():
        raise ConfigError(f"Damage calc script not found at {script_path}.")
    if not dependency_path.exists():
        raise ConfigError(
            f"{dependency_path} is missing. Run ./scripts/bootstrap-node-deps.sh first."
        )

    try:
        completed = subprocess.run(
            ["node", str(script_path)],
            cwd=project_root,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise ConfigError(
            f"Damage calc command timed out after {timeout_seconds} seconds for {payload_summary}."
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise ConfigError(
            f"Damage calc command exited with code {completed.returncode} for {payload_summary}: {detail}"
        )

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"Damage calc command returned invalid JSON: {completed.stdout.strip()}"
        ) from error

    if not isinstance(result, dict):
        raise ConfigError("Damage calc command returned a non-object JSON value.")
    _validate_schema(CALC_RESULT_SCHEMA_NAME, result)

    return result


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
