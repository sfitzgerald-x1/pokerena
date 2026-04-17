from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import List, Optional, Sequence

from .config import ConfigError, load_agents_config, load_server_config
from .showdown import build_server_command, node_version, prepare_runtime


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


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
    _add_common_config_arguments(up_parser, include_agents=False)
    up_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render runtime files and print the launch command without starting the process.",
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


def run_up(args: argparse.Namespace) -> int:
    server_config = load_server_config(config_path=args.config)
    artifacts = prepare_runtime(server_config)
    command = build_server_command(server_config)

    print(f"Runtime config: {artifacts.runtime_config_path}")
    print(f"Showdown config link: {artifacts.submodule_config_path}")
    print("Command:", " ".join(command))

    if args.dry_run:
        return 0

    if shutil.which("node") is None:
        print("error: node is required on PATH. Run scripts/bootstrap-showdown.sh first.", file=sys.stderr)
        return 1

    version = node_version()
    if not _is_supported_node_version(version):
        print(f"error: Node.js 22+ is required; found {version or 'no node installation'}.", file=sys.stderr)
        return 1

    if shutil.which("npm") is None:
        print("error: npm is required on PATH. Run scripts/bootstrap-showdown.sh first.", file=sys.stderr)
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
            f"error: {node_modules} is missing. Run scripts/bootstrap-showdown.sh first.",
            file=sys.stderr,
        )
        return 1

    completed = subprocess.run(command, cwd=server_config.showdown_path)
    return completed.returncode


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
                name="showdown-submodule",
                ok=launcher_path.exists(),
                detail=f"launcher expected at {launcher_path}",
            )
        )
        results.append(
            CheckResult(
                name="showdown-deps",
                ok=node_modules.exists(),
                detail=(
                    f"found {node_modules}"
                    if node_modules.exists()
                    else f"{node_modules} missing; run scripts/bootstrap-showdown.sh"
                ),
            )
        )

    return results


def _is_supported_node_version(version: Optional[str]) -> bool:
    if not version:
        return False
    normalized = version.lstrip("v")
    major = normalized.split(".", 1)[0]
    try:
        return int(major) >= 22
    except ValueError:
        return False
