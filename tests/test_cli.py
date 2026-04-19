import os
from io import StringIO
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest import mock

from pokerena.agent import BattleSession, save_capture
from pokerena.cli import collect_doctor_checks, main


class CLITest(unittest.TestCase):
    def test_render_config_command_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "vendor" / "pokemon-showdown" / "config").mkdir(parents=True)
            git_dir = root / ".git" / "modules" / "vendor" / "pokemon-showdown"
            (git_dir / "info").mkdir(parents=True)
            (root / "vendor" / "pokemon-showdown" / ".git").write_text(
                f"gitdir: {git_dir}\n",
                encoding="utf-8",
            )
            (root / "config" / "server.local.yaml").write_text(
                textwrap.dedent(
                    """
                    showdown_path: vendor/pokemon-showdown
                    bind_address: 0.0.0.0
                    port: 8000
                    server_id: pokerena-local
                    public_origin: http://localhost:8000
                    no_security: true
                    data_dir: .runtime/showdown/data
                    log_dir: .runtime/showdown/logs
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with mock.patch("sys.stdout", new=StringIO()) as buffer:
                old_cwd = Path.cwd()
                try:
                    os.chdir(root)
                    code = main(["server", "render-config", "--config", "config/server.local.yaml"])
                finally:
                    os.chdir(old_cwd)

            self.assertEqual(code, 0)
            self.assertIn("Rendered runtime config", buffer.getvalue())

    def test_up_dry_run_prints_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "vendor" / "pokemon-showdown" / "config").mkdir(parents=True)
            git_dir = root / ".git" / "modules" / "vendor" / "pokemon-showdown"
            (git_dir / "info").mkdir(parents=True)
            (root / "vendor" / "pokemon-showdown" / ".git").write_text(
                f"gitdir: {git_dir}\n",
                encoding="utf-8",
            )
            (root / "config" / "server.local.yaml").write_text(
                textwrap.dedent(
                    """
                    showdown_path: vendor/pokemon-showdown
                    bind_address: 0.0.0.0
                    port: 8000
                    server_id: pokerena-local
                    public_origin: http://localhost:8000
                    no_security: true
                    data_dir: .runtime/showdown/data
                    log_dir: .runtime/showdown/logs
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with mock.patch("sys.stdout", new=StringIO()) as buffer:
                old_cwd = Path.cwd()
                try:
                    os.chdir(root)
                    code = main(["server", "up", "--config", "config/server.local.yaml", "--dry-run"])
                finally:
                    os.chdir(old_cwd)

            self.assertEqual(code, 0)
            self.assertIn("node", buffer.getvalue())

    def test_doctor_reports_missing_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "vendor" / "pokemon-showdown").mkdir(parents=True)
            (root / "config" / "server.local.yaml").write_text(
                textwrap.dedent(
                    """
                    showdown_path: vendor/pokemon-showdown
                    bind_address: 0.0.0.0
                    port: 8000
                    server_id: pokerena-local
                    public_origin: http://localhost:8000
                    no_security: true
                    data_dir: .runtime/showdown/data
                    log_dir: .runtime/showdown/logs
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (root / "config" / "agents.yaml").write_text("agents: []\n", encoding="utf-8")

            with mock.patch("pokerena.cli.shutil.which", return_value=None):
                checks = collect_doctor_checks("config/server.local.yaml", "config/agents.yaml", root)

            self.assertTrue(any(check.name == "node" and not check.ok for check in checks))

    def test_agent_context_command_reads_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: test-agent
                        enabled: true
                        provider: codex
                        player_slot: p1
                        format_allowlist: [gen9randombattle]
                        transport: sim-stream
                        launch:
                          command: cat
                          args: []
                          cwd: .
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            session = BattleSession(battle_id="battle-cli", player_slot="p1")
            session.ingest(
                _public_event(
                    "battle-cli",
                    ["|tier|[Gen 9] Random Battle", "|turn|1"],
                )
            )
            session.ingest(
                _request_event(
                    "battle-cli",
                    "p1",
                    {
                        "active": [{"moves": [{"move": "Moonblast", "id": "moonblast", "disabled": False}]}],
                        "side": {"pokemon": [{"ident": "p1: Flutter Mane", "condition": "100/100", "active": True}]},
                    },
                )
            )
            capture_path = root / "battle-capture.json"
            save_capture(capture_path, session.to_capture())

            with mock.patch("sys.stdout", new=StringIO()) as buffer:
                old_cwd = Path.cwd()
                try:
                    os.chdir(root)
                    code = main(
                        [
                            "agent",
                            "context",
                            "--agent-id",
                            "test-agent",
                            "--agents-config",
                            "config/agents.yaml",
                            "--capture",
                            str(capture_path),
                        ]
                    )
                finally:
                    os.chdir(old_cwd)

            self.assertEqual(code, 0)
            self.assertIn('"schema_version": "pokerena.turn-context.v1"', buffer.getvalue())


def _public_event(battle_id: str, lines: list[str]):
    from pokerena.agent import SessionEvent

    return SessionEvent(event_type="public_update", battle_id=battle_id, payload={"lines": lines})


def _request_event(battle_id: str, player_slot: str, payload: dict):
    from pokerena.agent import SessionEvent

    return SessionEvent(
        event_type="request_received",
        battle_id=battle_id,
        player_slot=player_slot,
        payload=payload,
    )
