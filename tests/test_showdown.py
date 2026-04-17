from pathlib import Path
import tempfile
import unittest

from pokerena.config import ServerConfig
from pokerena.showdown import build_server_command, prepare_runtime, render_showdown_config


class ShowdownRuntimeTest(unittest.TestCase):
    def test_render_showdown_config_contains_expected_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = ServerConfig(
                project_root=root,
                config_path=root / "config" / "server.local.yaml",
                showdown_path=root / "vendor" / "pokemon-showdown",
                bind_address="0.0.0.0",
                port=8000,
                server_id="pokerena-local",
                public_origin="http://localhost:8000",
                no_security=True,
                data_dir=root / ".runtime" / "showdown" / "data",
                log_dir=root / ".runtime" / "showdown" / "logs",
                runtime_dir=root / ".runtime" / "showdown",
            )

            rendered = render_showdown_config(config)

            self.assertIn("exports.port = 8000;", rendered)
            self.assertIn("exports.serverid = \"pokerena-local\";", rendered)
            self.assertIn("publicOrigin", rendered)

    def test_prepare_runtime_writes_files_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            showdown_path = root / "vendor" / "pokemon-showdown"
            (showdown_path / "config").mkdir(parents=True)
            git_dir = root / ".git" / "modules" / "vendor" / "pokemon-showdown"
            (git_dir / "info").mkdir(parents=True)
            (showdown_path / ".git").write_text(
                f"gitdir: {git_dir}\n",
                encoding="utf-8",
            )
            config = ServerConfig(
                project_root=root,
                config_path=root / "config" / "server.local.yaml",
                showdown_path=showdown_path,
                bind_address="0.0.0.0",
                port=8000,
                server_id="pokerena-local",
                public_origin="http://localhost:8000",
                no_security=False,
                data_dir=root / ".runtime" / "showdown" / "data",
                log_dir=root / ".runtime" / "showdown" / "logs",
                runtime_dir=root / ".runtime" / "showdown",
            )

            artifacts = prepare_runtime(config)

            self.assertTrue(artifacts.runtime_config_path.exists())
            self.assertTrue(artifacts.submodule_config_path.is_symlink())
            self.assertTrue(artifacts.runtime_metadata_path.exists())

    def test_build_server_command_adds_no_security_flag(self) -> None:
        config = ServerConfig(
            project_root=Path("/tmp/project"),
            config_path=Path("/tmp/project/config/server.local.yaml"),
            showdown_path=Path("/tmp/project/vendor/pokemon-showdown"),
            bind_address="0.0.0.0",
            port=8000,
            server_id="pokerena-local",
            public_origin="http://localhost:8000",
            no_security=True,
            data_dir=Path("/tmp/project/.runtime/showdown/data"),
            log_dir=Path("/tmp/project/.runtime/showdown/logs"),
            runtime_dir=Path("/tmp/project/.runtime/showdown"),
        )

        command = build_server_command(config)

        self.assertEqual(command[:3], ["node", "/tmp/project/vendor/pokemon-showdown/pokemon-showdown", "start"])
        self.assertEqual(command[-1], "--no-security")

