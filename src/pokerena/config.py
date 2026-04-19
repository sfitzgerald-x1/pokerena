from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import ipaddress
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
    transcript_viewer: "TranscriptViewerConfig" = field(default_factory=lambda: TranscriptViewerConfig())


@dataclass(frozen=True)
class AgentLaunchConfig:
    command: str
    args: List[str]
    cwd: Path


@dataclass(frozen=True)
class AgentHookConfig:
    type: str
    context_format: str
    decision_format: str
    prompt_style: str
    history_turn_limit: int = 4
    decision_timeout_seconds: int = 120


@dataclass(frozen=True)
class TranscriptViewerConfig:
    enabled: bool = True
    bind_address: str = "127.0.0.1"
    port: int = 8001


@dataclass(frozen=True)
class AgentCallableConfig:
    enabled: bool
    username: Optional[str]
    accepted_formats: List[str]
    challenge_policy: str
    avatar: Optional[str]


@dataclass(frozen=True)
class AgentPricingConfig:
    model: Optional[str] = None
    input_usd_per_million_tokens: Optional[Decimal] = None
    output_usd_per_million_tokens: Optional[Decimal] = None
    cache_read_input_usd_per_million_tokens: Optional[Decimal] = None
    cache_creation_input_usd_per_million_tokens: Optional[Decimal] = None


@dataclass(frozen=True)
class AgentDefinition:
    agent_id: str
    enabled: bool
    provider: str
    player_slot: str
    format_allowlist: List[str]
    transport: str
    launch: AgentLaunchConfig
    hook: AgentHookConfig
    env_file: Optional[Path]
    callable: AgentCallableConfig
    pricing: AgentPricingConfig


ENVIRONMENT_OVERRIDES = {
    "POKERENA_SHOWDOWN_PATH": "showdown_path",
    "POKERENA_BIND_ADDRESS": "bind_address",
    "POKERENA_PORT": "port",
    "POKERENA_SERVER_ID": "server_id",
    "POKERENA_PUBLIC_ORIGIN": "public_origin",
    "POKERENA_NO_SECURITY": "no_security",
    "POKERENA_DATA_DIR": "data_dir",
    "POKERENA_LOG_DIR": "log_dir",
    "POKERENA_TRANSCRIPT_VIEWER_ENABLED": "transcript_viewer.enabled",
    "POKERENA_TRANSCRIPT_VIEWER_BIND_ADDRESS": "transcript_viewer.bind_address",
    "POKERENA_TRANSCRIPT_VIEWER_PORT": "transcript_viewer.port",
}

ALLOWED_AGENT_TRANSPORTS = {"sim-stream", "showdown-client"}
ALLOWED_CONTEXT_FORMATS = {"pokerena.turn-context.v1"}
ALLOWED_DECISION_FORMATS = {"pokerena.decision.v1"}
ALLOWED_PROMPT_STYLES = {"showdown-turn-v1"}
ALLOWED_CHALLENGE_POLICIES = {"accept-direct-challenges"}


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
    transcript_viewer_raw = values.get("transcript_viewer", {})
    if transcript_viewer_raw is None:
        transcript_viewer_raw = {}
    if not isinstance(transcript_viewer_raw, dict):
        raise ConfigError("config.transcript_viewer must be a mapping.")
    transcript_viewer = TranscriptViewerConfig(
        enabled=_parse_bool(_nested_value(values, "transcript_viewer.enabled", transcript_viewer_raw.get("enabled", True))),
        bind_address=_require_loopback_bind_address(
            _required_string(
                {"bind_address": _nested_value(values, "transcript_viewer.bind_address", transcript_viewer_raw.get("bind_address", "127.0.0.1"))},
                "bind_address",
                prefix="config.transcript_viewer",
            ),
            key="config.transcript_viewer.bind_address",
        ),
        port=_parse_port(_nested_value(values, "transcript_viewer.port", transcript_viewer_raw.get("port", 8001))),
    )

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
        transcript_viewer=transcript_viewer,
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

        hook = raw_agent.get("hook", {})
        if hook is None:
            hook = {}
        if not isinstance(hook, dict):
            raise ConfigError(f"agents[{index}].hook must be a mapping.")

        context_format = _optional_string(
            hook.get("context_format"),
            default="pokerena.turn-context.v1",
        )
        if context_format not in ALLOWED_CONTEXT_FORMATS:
            allowed = ", ".join(sorted(ALLOWED_CONTEXT_FORMATS))
            raise ConfigError(f"agents[{index}].hook.context_format must be one of: {allowed}.")

        decision_format = _optional_string(
            hook.get("decision_format"),
            default="pokerena.decision.v1",
        )
        if decision_format not in ALLOWED_DECISION_FORMATS:
            allowed = ", ".join(sorted(ALLOWED_DECISION_FORMATS))
            raise ConfigError(f"agents[{index}].hook.decision_format must be one of: {allowed}.")

        prompt_style = _optional_string(
            hook.get("prompt_style"),
            default="showdown-turn-v1",
        )
        if prompt_style not in ALLOWED_PROMPT_STYLES:
            allowed = ", ".join(sorted(ALLOWED_PROMPT_STYLES))
            raise ConfigError(f"agents[{index}].hook.prompt_style must be one of: {allowed}.")
        history_turn_limit = _parse_positive_int(
            hook.get("history_turn_limit", 4),
            key=f"agents[{index}].hook.history_turn_limit",
        )
        decision_timeout_seconds = _parse_positive_int(
            hook.get("decision_timeout_seconds", 120),
            key=f"agents[{index}].hook.decision_timeout_seconds",
        )

        env_file_value = raw_agent.get("env_file")
        env_file = None
        if env_file_value is not None:
            if not isinstance(env_file_value, str) or not env_file_value.strip():
                raise ConfigError(f"agents[{index}].env_file must be a non-empty string if set.")
            env_file = _resolve_project_path(root, env_file_value)

        callable_block = raw_agent.get("callable", {})
        if callable_block is None:
            callable_block = {}
        if not isinstance(callable_block, dict):
            raise ConfigError(f"agents[{index}].callable must be a mapping.")

        callable_enabled = _parse_bool(callable_block.get("enabled", False))
        callable_username = None
        username_value = callable_block.get("username")
        if username_value is not None:
            callable_username = _optional_string(username_value, default="")
            if not callable_username:
                raise ConfigError(f"agents[{index}].callable.username must be a non-empty string if set.")

        accepted_formats_value = callable_block.get("accepted_formats")
        if accepted_formats_value is None:
            accepted_formats = list(allowlist)
        else:
            if (
                not isinstance(accepted_formats_value, list)
                or any(not isinstance(item, str) or not item.strip() for item in accepted_formats_value)
            ):
                raise ConfigError(f"agents[{index}].callable.accepted_formats must be a list of strings.")
            accepted_formats = [item.strip() for item in accepted_formats_value]

        challenge_policy = _optional_string(
            callable_block.get("challenge_policy"),
            default="accept-direct-challenges",
        )
        if challenge_policy not in ALLOWED_CHALLENGE_POLICIES:
            allowed = ", ".join(sorted(ALLOWED_CHALLENGE_POLICIES))
            raise ConfigError(f"agents[{index}].callable.challenge_policy must be one of: {allowed}.")

        avatar = None
        avatar_value = callable_block.get("avatar")
        if avatar_value is not None:
            avatar = _optional_string(avatar_value, default="")
            if not avatar:
                raise ConfigError(f"agents[{index}].callable.avatar must be a non-empty string if set.")
            if not _is_safe_showdown_avatar(avatar):
                raise ConfigError(
                    f"agents[{index}].callable.avatar must contain only letters, numbers, underscores, or hyphens."
                )

        pricing_block = raw_agent.get("pricing", {})
        if pricing_block is None:
            pricing_block = {}
        if not isinstance(pricing_block, dict):
            raise ConfigError(f"agents[{index}].pricing must be a mapping.")

        pricing_model = None
        pricing_model_value = pricing_block.get("model")
        if pricing_model_value is not None:
            pricing_model = _optional_string(pricing_model_value, default="")
            if not pricing_model:
                raise ConfigError(f"agents[{index}].pricing.model must be a non-empty string if set.")

        transport = _required_string(raw_agent, "transport", prefix=f"agents[{index}]")
        if transport not in ALLOWED_AGENT_TRANSPORTS:
            allowed = ", ".join(sorted(ALLOWED_AGENT_TRANSPORTS))
            raise ConfigError(
                f"agents[{index}].transport must be one of: {allowed}."
            )

        if callable_enabled:
            if transport != "showdown-client":
                raise ConfigError(
                    f"agents[{index}].callable.enabled requires transport showdown-client."
                )
            if not callable_username:
                raise ConfigError(
                    f"agents[{index}].callable.username is required when callable.enabled is true."
                )
            if not accepted_formats:
                raise ConfigError(
                    f"agents[{index}].callable.accepted_formats must not be empty when callable.enabled is true."
                )

        agents.append(
            AgentDefinition(
                agent_id=_required_string(raw_agent, "id", prefix=f"agents[{index}]"),
                enabled=_parse_bool(raw_agent.get("enabled", True)),
                provider=_optional_string(raw_agent.get("provider"), default="generic"),
                player_slot=_optional_string(raw_agent.get("player_slot"), default="p1"),
                format_allowlist=allowlist,
                transport=transport,
                launch=AgentLaunchConfig(command=command, args=list(args), cwd=launch_cwd),
                hook=AgentHookConfig(
                    type=_optional_string(hook.get("type"), default="subprocess_stdio"),
                    context_format=context_format,
                    decision_format=decision_format,
                    prompt_style=prompt_style,
                    history_turn_limit=history_turn_limit,
                    decision_timeout_seconds=decision_timeout_seconds,
                ),
                env_file=env_file,
                callable=AgentCallableConfig(
                    enabled=callable_enabled,
                    username=callable_username,
                    accepted_formats=accepted_formats,
                    challenge_policy=challenge_policy,
                    avatar=avatar,
                ),
                pricing=AgentPricingConfig(
                    model=pricing_model,
                    input_usd_per_million_tokens=_parse_optional_non_negative_decimal(
                        pricing_block.get("input_usd_per_million_tokens"),
                        key=f"agents[{index}].pricing.input_usd_per_million_tokens",
                    ),
                    output_usd_per_million_tokens=_parse_optional_non_negative_decimal(
                        pricing_block.get("output_usd_per_million_tokens"),
                        key=f"agents[{index}].pricing.output_usd_per_million_tokens",
                    ),
                    cache_read_input_usd_per_million_tokens=_parse_optional_non_negative_decimal(
                        pricing_block.get("cache_read_input_usd_per_million_tokens"),
                        key=f"agents[{index}].pricing.cache_read_input_usd_per_million_tokens",
                    ),
                    cache_creation_input_usd_per_million_tokens=_parse_optional_non_negative_decimal(
                        pricing_block.get("cache_creation_input_usd_per_million_tokens"),
                        key=f"agents[{index}].pricing.cache_creation_input_usd_per_million_tokens",
                    ),
                ),
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


def _optional_string(value: object, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Expected a non-empty string value, got {value!r}.")
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


def _parse_positive_int(value: object, *, key: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{key} must be a positive integer.") from error
    if isinstance(value, bool) or parsed < 1:
        raise ConfigError(f"{key} must be a positive integer.")
    return parsed


def _parse_optional_non_negative_decimal(value: object, *, key: str) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"{key} must be a non-negative number.")
    if not isinstance(value, (int, float, str)):
        raise ConfigError(f"{key} must be a non-negative number.")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ConfigError(f"{key} must be a non-negative number.") from error
    if parsed < 0:
        raise ConfigError(f"{key} must be a non-negative number.")
    return parsed


def _require_loopback_bind_address(value: str, *, key: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ConfigError(f"{key} must be a non-empty string.")
    if normalized.lower() == "localhost":
        return normalized
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError as error:
        raise ConfigError(
            f"{key} must stay on a loopback address such as 127.0.0.1, localhost, or ::1."
        ) from error
    if not address.is_loopback:
        raise ConfigError(
            f"{key} must stay on a loopback address such as 127.0.0.1, localhost, or ::1."
        )
    return normalized


def _is_safe_showdown_avatar(value: str) -> bool:
    return all(character.isalnum() or character in {"_", "-"} for character in value)


def _nested_value(values: Dict[str, object], key: str, default: object) -> object:
    return values.get(key, default)
