from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

from .config import ConfigError


CALC_SCRIPT_PATH = Path("tools") / "damage-calc-cli.cjs"
CALC_DEPENDENCY_PATH = Path("node_modules") / "@smogon" / "calc"


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

    _require_positive_int(payload.get("generation"), "generation")
    _require_named_object(payload.get("attacker"), "attacker", "species")
    _require_named_object(payload.get("defender"), "defender", "species")
    _require_named_object(payload.get("move"), "move", "name")
    _require_optional_object(payload.get("field"), "field")

    return payload


def run_damage_calc(payload: Dict[str, Any], *, project_root: Path) -> Dict[str, Any]:
    script_path = project_root / CALC_SCRIPT_PATH
    dependency_path = project_root / CALC_DEPENDENCY_PATH

    if shutil.which("node") is None:
        raise ConfigError("node is required on PATH. Run ./scripts/bootstrap-node-deps.sh first.")
    if not script_path.exists():
        raise ConfigError(f"Damage calc script not found at {script_path}.")
    if not dependency_path.exists():
        raise ConfigError(
            f"{dependency_path} is missing. Run ./scripts/bootstrap-node-deps.sh first."
        )

    completed = subprocess.run(
        ["node", str(script_path)],
        cwd=project_root,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise ConfigError(
            f"Damage calc command exited with code {completed.returncode}: {detail}"
        )

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"Damage calc command returned invalid JSON: {completed.stdout.strip()}"
        ) from error

    if not isinstance(result, dict):
        raise ConfigError("Damage calc command returned a non-object JSON value.")

    return result


def sample_damage_calc_payload() -> Dict[str, Any]:
    return {
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


def _require_positive_int(value: Any, label: str) -> None:
    if not isinstance(value, int) or value < 1:
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
