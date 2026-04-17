from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Dict, List, Optional

import yaml


class ConfigError(ValueError):
    """Raised when a Pokerena config file is invalid."""


@dataclass(frozen=True)
class ServerConfig:
    project_root: Path
    config_path: Path
    showdown_path: Path
    bind_address: str
    port: int
    server_id: str
    public_origin: str
    no_security: bool
    data_dir: Path
    log_dir: Path
    runtime_dir: Path


@dataclass(frozen=True)
class AgentLaunchConfig:
    command: str
    args: List[str]
    cwd: Path


@dataclass(frozen=True)
class AgentDefinition:
    agent_id: str
    enabled: bool
    format_allowlist: List[str]
    transport: str
    launch: AgentLaunchConfig
    env_file: Optional[Path]


ENVIRONMENT_OVERRIDES = {
    "POKERENA_SHOWDOWN_PATH": "showdown_path",
    "POKERENA_BIND_ADDRESS": "bind_address",
    "POKERENA_PORT": "port",
    "POKERENA_SERVER_ID": "server_id",
    "POKERENA_PUBLIC_ORIGIN": "public_origin",
    "POKERENA_NO_SECURITY": "no_security",
    "POKERENA_DATA_DIR": "data_dir",
    "POKERENA_LOG_DIR": "log_dir",
}


def load_server_config(
    config_path: str = "config/server.local.yaml",
    project_root: Optional[Path] = None,
) -> ServerConfig:
    root = _resolve_project_root(project_root)
    resolved_path = _resolve_input_path(root, config_path)
    if not resolved_path.exists():
        raise ConfigError(
            f"Server config not found at {resolved_path}. Copy config/server.local.example.yaml first."
        )

    raw = _load_yaml_mapping(resolved_path)
    env = _load_env_overrides(root)
    values = dict(raw)
    for env_name, key in ENVIRONMENT_OVERRIDES.items():
        if env_name in env:
            values[key] = env[env_name]

    showdown_path = _resolve_project_path(root, _required_string(values, "showdown_path"))
    bind_address = _required_string(values, "bind_address")
    port = _parse_port(values.get("port"))
    server_id = _required_string(values, "server_id")
    public_origin = _required_string(values, "public_origin")
    no_security = _parse_bool(values.get("no_security", False))
    data_dir = _resolve_project_path(root, _required_string(values, "data_dir"))
    log_dir = _resolve_project_path(root, _required_string(values, "log_dir"))
    runtime_dir = root / ".runtime" / "showdown"

    return ServerConfig(
        project_root=root,
        config_path=resolved_path,
        showdown_path=showdown_path,
        bind_address=bind_address,
        port=port,
        server_id=server_id,
        public_origin=public_origin,
        no_security=no_security,
        data_dir=data_dir,
        log_dir=log_dir,
        runtime_dir=runtime_dir,
    )


def load_agents_config(
    config_path: str = "config/agents.yaml",
    project_root: Optional[Path] = None,
) -> List[AgentDefinition]:
    root = _resolve_project_root(project_root)
    resolved_path = _resolve_input_path(root, config_path)
    if not resolved_path.exists():
        raise ConfigError(
            f"Agents config not found at {resolved_path}. Copy config/agents.example.yaml first."
        )

    raw = _load_yaml_mapping(resolved_path)
    raw_agents = raw.get("agents", [])
    if not isinstance(raw_agents, list):
        raise ConfigError("`agents` must be a list.")

    agents: List[AgentDefinition] = []
    for index, raw_agent in enumerate(raw_agents):
        if not isinstance(raw_agent, dict):
            raise ConfigError(f"agents[{index}] must be a mapping.")

        launch = raw_agent.get("launch")
        if not isinstance(launch, dict):
            raise ConfigError(f"agents[{index}].launch must be a mapping.")

        command = _required_string(launch, "command", prefix=f"agents[{index}].launch")
        args = launch.get("args", [])
        if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
            raise ConfigError(f"agents[{index}].launch.args must be a list of strings.")

        launch_cwd = _resolve_project_path(
            root,
            _required_string(launch, "cwd", prefix=f"agents[{index}].launch"),
        )
        allowlist = raw_agent.get("format_allowlist", [])
        if not isinstance(allowlist, list) or any(not isinstance(item, str) for item in allowlist):
            raise ConfigError(f"agents[{index}].format_allowlist must be a list of strings.")

        env_file_value = raw_agent.get("env_file")
        env_file = None
        if env_file_value is not None:
            if not isinstance(env_file_value, str) or not env_file_value.strip():
                raise ConfigError(f"agents[{index}].env_file must be a non-empty string if set.")
            env_file = _resolve_project_path(root, env_file_value)

        agents.append(
            AgentDefinition(
                agent_id=_required_string(raw_agent, "id", prefix=f"agents[{index}]"),
                enabled=_parse_bool(raw_agent.get("enabled", True)),
                format_allowlist=allowlist,
                transport=_required_string(raw_agent, "transport", prefix=f"agents[{index}]"),
                launch=AgentLaunchConfig(command=command, args=list(args), cwd=launch_cwd),
                env_file=env_file,
            )
        )

    return agents


def _load_env_overrides(project_root: Path) -> Dict[str, str]:
    env = {}
    env_path = project_root / ".env"
    if env_path.exists():
        env.update(_parse_dotenv(env_path))
    env.update(os.environ)
    return env


def _parse_dotenv(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            raise ConfigError(f"Invalid .env line in {path}: {raw_line}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _load_yaml_mapping(path: Path) -> Dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a top-level mapping.")
    return loaded


def _resolve_project_root(project_root: Optional[Path]) -> Path:
    return Path(project_root or Path.cwd()).resolve()


def _resolve_input_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _required_string(
    values: Dict[str, object],
    key: str,
    prefix: str = "config",
) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{prefix}.{key} must be a non-empty string.")
    return value.strip()


def _parse_port(value: object) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError("config.port must be an integer.") from error
    if port < 1 or port > 65535:
        raise ConfigError("config.port must be between 1 and 65535.")
    return port


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ConfigError(f"Expected a boolean value, got {value!r}.")

