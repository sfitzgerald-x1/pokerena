import os
from io import StringIO
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest import mock

from pokerena.agent import AgentContextCursor, BattleSession, DecisionPolicy, save_capture
from pokerena.cli import _decide_or_fallback, collect_doctor_checks, main
from pokerena.config import load_agents_config


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

    def test_sim_battle_rejects_disabled_agent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "vendor" / "pokemon-showdown").mkdir(parents=True)
            (root / "config" / "server.local.yaml").write_text(
                textwrap.dedent(
                    """
                    showdown_path: vendor/pokemon-showdown
                    bind_address: 127.0.0.1
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
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: disabled-agent
                        enabled: false
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

            with mock.patch("sys.stderr", new=StringIO()) as buffer:
                old_cwd = Path.cwd()
                try:
                    os.chdir(root)
                    code = main(
                        [
                            "agent",
                            "sim-battle",
                            "--config",
                            "config/server.local.yaml",
                            "--agents-config",
                            "config/agents.yaml",
                            "--agent-id",
                            "disabled-agent",
                        ]
                    )
                finally:
                    os.chdir(old_cwd)

            self.assertEqual(code, 2)
            self.assertIn("disabled", buffer.getvalue())

    def test_retry_limit_uses_fallback_without_invoking_hook(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: retry-agent
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
            agent = load_agents_config(project_root=root)[0]
            session = BattleSession(battle_id="battle-retry", player_slot="p1")
            session.ingest(_public_event("battle-retry", ["|turn|4"]))
            session.ingest(
                _request_event(
                    "battle-retry",
                    "p1",
                    {
                        "active": [{"moves": [{"move": "Thunderbolt", "id": "thunderbolt", "disabled": False}]}],
                        "side": {"pokemon": [{"ident": "p1: Pikachu", "condition": "100/100", "active": True}]},
                    },
                )
            )
            session.ingest(_rejection_event("battle-retry", "p1", "[Unavailable choice] No"))
            session.ingest(_rejection_event("battle-retry", "p1", "[Unavailable choice] Still no"))
            session.ingest(_rejection_event("battle-retry", "p1", "[Unavailable choice] Last no"))
            context = session.build_turn_context(
                agent=agent,
                cursor=AgentContextCursor(last_turn_number=3, last_request_sequence=0),
            )

            with mock.patch("pokerena.cli.invoke_agent") as invoke_agent:
                decision = _decide_or_fallback(
                    agent=agent,
                    session=session,
                    context=context,
                    runtime_root=root / ".runtime",
                    capture_path=root / "capture.json",
                    policy=DecisionPolicy(max_invalid_retries=3, decision_timeout_seconds=10, history_limit=60),
                )

            self.assertEqual(decision, "move 1")
            invoke_agent.assert_not_called()

    def test_sim_battle_rejects_format_outside_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "vendor" / "pokemon-showdown").mkdir(parents=True)
            (root / "config" / "server.local.yaml").write_text(
                textwrap.dedent(
                    """
                    showdown_path: vendor/pokemon-showdown
                    bind_address: 127.0.0.1
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
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: test-agent
                        enabled: true
                        provider: codex
                        player_slot: p1
                        format_allowlist: [gen1randombattle]
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

            with mock.patch("sys.stderr", new=StringIO()) as buffer:
                old_cwd = Path.cwd()
                try:
                    os.chdir(root)
                    code = main(
                        [
                            "agent",
                            "sim-battle",
                            "--config",
                            "config/server.local.yaml",
                            "--agents-config",
                            "config/agents.yaml",
                            "--agent-id",
                            "test-agent",
                            "--format",
                            "gen9randombattle",
                        ]
                    )
                finally:
                    os.chdir(old_cwd)

            self.assertEqual(code, 2)
            self.assertIn("not allowed", buffer.getvalue())


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


def _rejection_event(battle_id: str, player_slot: str, message: str):
    from pokerena.agent import SessionEvent

    return SessionEvent(
        event_type="choice_rejected",
        battle_id=battle_id,
        player_slot=player_slot,
        payload={"message": message},
    )
