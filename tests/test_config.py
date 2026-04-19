import os
from pathlib import Path
import tempfile
import textwrap
import unittest

from pokerena.config import ConfigError, load_agents_config, load_server_config


class ConfigLoadingTest(unittest.TestCase):
    def test_load_server_config_uses_yaml_and_env_overrides(self) -> None:
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
                    no_security: false
                    data_dir: .runtime/showdown/data
                    log_dir: .runtime/showdown/logs
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (root / ".env").write_text("POKERENA_PORT=9000\nPOKERENA_NO_SECURITY=true\n", encoding="utf-8")

            old_cwd = Path.cwd()
            old_environ = dict(os.environ)
            try:
                os.chdir(root)
                os.environ["POKERENA_SERVER_ID"] = "env-server"
                config = load_server_config()
            finally:
                os.chdir(old_cwd)
                os.environ.clear()
                os.environ.update(old_environ)

            self.assertEqual(config.port, 9000)
            self.assertTrue(config.no_security)
            self.assertEqual(config.server_id, "env-server")
            self.assertTrue(config.transcript_viewer.enabled)
            self.assertEqual(config.transcript_viewer.port, 8001)
            self.assertEqual(
                config.showdown_path.resolve(),
                (root / "vendor" / "pokemon-showdown").resolve(),
            )

    def test_load_agents_config_validates_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: test-bot
                        enabled: true
                        format_allowlist:
                          - gen9randombattle
                        transport: sim-stream
                        launch:
                          command: python3.14
                          args: ["-m", "bot"]
                          cwd: .
                        env_file: .env
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            agents = load_agents_config(project_root=root)

            self.assertEqual(len(agents), 1)
            self.assertEqual(agents[0].agent_id, "test-bot")
            self.assertEqual(agents[0].launch.command, "python3.14")
            self.assertEqual(agents[0].transport, "sim-stream")
            self.assertEqual(agents[0].hook.context_format, "pokerena.turn-context.v1")
            self.assertEqual(agents[0].hook.history_turn_limit, 4)
            self.assertEqual(agents[0].hook.decision_timeout_seconds, 120)
            self.assertFalse(agents[0].callable.enabled)

    def test_load_server_config_accepts_transcript_viewer_block(self) -> None:
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
                    transcript_viewer:
                      enabled: true
                      bind_address: 127.0.0.1
                      port: 8123
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_server_config(project_root=root)

            self.assertTrue(config.transcript_viewer.enabled)
            self.assertEqual(config.transcript_viewer.port, 8123)

    def test_load_agents_config_accepts_hook_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: test-bot
                        enabled: true
                        provider: claude
                        player_slot: p1
                        format_allowlist: [gen3randombattle]
                        transport: showdown-client
                        launch:
                          command: claude
                          args: ["-p"]
                          cwd: .
                        hook:
                          history_turn_limit: 12
                          decision_timeout_seconds: 180
                        callable:
                          enabled: true
                          username: ClaudeLocalBot
                          accepted_formats: [gen3randombattle]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            agents = load_agents_config(project_root=root)

            self.assertEqual(agents[0].hook.history_turn_limit, 12)
            self.assertEqual(agents[0].hook.decision_timeout_seconds, 180)

    def test_load_agents_config_validates_callable_agents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: local-bot
                        enabled: true
                        provider: claude
                        player_slot: p1
                        format_allowlist: [gen3randombattle]
                        transport: showdown-client
                        launch:
                          command: python3.14
                          args: ["-m", "bot"]
                          cwd: .
                        callable:
                          enabled: true
                          username: ClaudeLocalBot
                          accepted_formats: [gen3randombattle]
                          challenge_policy: accept-direct-challenges
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            agents = load_agents_config(project_root=root)

            self.assertTrue(agents[0].callable.enabled)
            self.assertEqual(agents[0].callable.username, "ClaudeLocalBot")
            self.assertEqual(agents[0].callable.accepted_formats, ["gen3randombattle"])

    def test_load_agents_config_rejects_callable_non_showdown_transport(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: invalid-bot
                        enabled: true
                        provider: claude
                        player_slot: p1
                        format_allowlist: [gen3randombattle]
                        transport: sim-stream
                        launch:
                          command: python3.14
                          args: ["-m", "bot"]
                          cwd: .
                        callable:
                          enabled: true
                          username: ClaudeLocalBot
                          accepted_formats: [gen3randombattle]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ConfigError):
                load_agents_config(project_root=root)

    def test_load_server_config_requires_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(ConfigError):
                load_server_config(project_root=root)
