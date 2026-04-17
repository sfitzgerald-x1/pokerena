from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import List, Optional

from .config import ServerConfig


@dataclass(frozen=True)
class RuntimeArtifacts:
    runtime_dir: Path
    runtime_config_path: Path
    runtime_metadata_path: Path
    submodule_config_path: Path


DEFAULT_ROUTES = {
    "root": "pokemonshowdown.com",
    "client": "play.pokemonshowdown.com",
    "dex": "dex.pokemonshowdown.com",
    "replays": "replay.pokemonshowdown.com",
}


def render_showdown_config(config: ServerConfig) -> str:
    lines = [
        "'use strict';",
        "",
        f"exports.port = {config.port};",
        f"exports.bindaddress = {json.dumps(config.bind_address)};",
        f"exports.serverid = {json.dumps(config.server_id)};",
        "exports.loginserver = 'http://play.pokemonshowdown.com/';",
        "exports.loginserverkeyalgo = 'RSA-SHA1';",
        "exports.loginserverpublickeyid = 4;",
        "exports.routes = {",
    ]
    for key, value in DEFAULT_ROUTES.items():
        lines.append(f"  {key}: {json.dumps(value)},")
    lines.extend(
        [
            "};",
            "exports.watchconfig = false;",
            "exports.repl = false;",
            "exports.logchat = false;",
            "exports.loguserstats = 0;",
            "exports.reportjoins = false;",
            "exports.reportbattlejoins = false;",
            "exports.pokerena = {",
            f"  publicOrigin: {json.dumps(config.public_origin)},",
            f"  dataDir: {json.dumps(str(config.data_dir))},",
            f"  logDir: {json.dumps(str(config.log_dir))},",
            "};",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_runtime(config: ServerConfig) -> RuntimeArtifacts:
    runtime_config_dir = config.runtime_dir / "config"
    runtime_config_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    runtime_config_path = runtime_config_dir / "config.js"
    runtime_config_path.write_text(render_showdown_config(config), encoding="utf-8")

    submodule_config_path = config.showdown_path / "config" / "config.js"
    _ensure_submodule_ignore(config.showdown_path, "config/config.js")
    _link_generated_config(runtime_config_path, submodule_config_path)

    metadata_path = config.runtime_dir / "server-runtime.json"
    metadata = {
        "showdown_path": str(config.showdown_path),
        "server_id": config.server_id,
        "public_origin": config.public_origin,
        "bind_address": config.bind_address,
        "port": config.port,
        "no_security": config.no_security,
        "data_dir": str(config.data_dir),
        "log_dir": str(config.log_dir),
        "runtime_config_path": str(runtime_config_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return RuntimeArtifacts(
        runtime_dir=config.runtime_dir,
        runtime_config_path=runtime_config_path,
        runtime_metadata_path=metadata_path,
        submodule_config_path=submodule_config_path,
    )


def build_server_command(config: ServerConfig) -> List[str]:
    command = ["node", str(config.showdown_path / "pokemon-showdown"), "start"]
    if config.no_security:
        command.append("--no-security")
    return command


def detect_git_dir(repo_path: Path) -> Optional[Path]:
    dot_git = repo_path / ".git"
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():
        raw = dot_git.read_text(encoding="utf-8").strip()
        prefix = "gitdir: "
        if raw.startswith(prefix):
            return (repo_path / raw[len(prefix) :]).resolve()
    return None


def node_version() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["node", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _link_generated_config(runtime_config_path: Path, submodule_config_path: Path) -> None:
    submodule_config_path.parent.mkdir(parents=True, exist_ok=True)
    if submodule_config_path.exists() or submodule_config_path.is_symlink():
        if submodule_config_path.is_symlink() and submodule_config_path.resolve() == runtime_config_path.resolve():
            return
        submodule_config_path.unlink()

    relative_target = Path(os.path.relpath(runtime_config_path, start=submodule_config_path.parent))
    submodule_config_path.symlink_to(relative_target)


def _ensure_submodule_ignore(showdown_path: Path, ignore_entry: str) -> None:
    git_dir = detect_git_dir(showdown_path)
    if not git_dir:
        return

    exclude_path = git_dir / "info" / "exclude"
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    if exclude_path.exists():
        existing = exclude_path.read_text(encoding="utf-8").splitlines()
    else:
        existing = []
    if ignore_entry not in existing:
        content = "\n".join(existing + [ignore_entry]).strip() + "\n"
        exclude_path.write_text(content, encoding="utf-8")
