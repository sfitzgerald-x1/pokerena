import json
import os
from io import StringIO
from pathlib import Path
import subprocess
import shutil
import tempfile
import textwrap
import unittest
from unittest import mock

from pokerena.calc import (
    CALC_REQUEST_SCHEMA_VERSION,
    detect_project_root,
    read_damage_calc_input,
    run_damage_calc,
    sample_damage_calc_payload,
)
from pokerena.cli import collect_doctor_checks, main
from pokerena.config import ConfigError


class DamageCalcTest(unittest.TestCase):
    def test_read_damage_calc_input_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload_path = Path(temp_dir) / "request.json"
            payload_path.write_text(json.dumps(sample_damage_calc_payload()), encoding="utf-8")

            payload = read_damage_calc_input(input_path=str(payload_path), use_stdin=False)

            self.assertEqual(payload["generation"], 2)
            self.assertEqual(payload["attacker"]["species"], "Snorlax")

    def test_read_damage_calc_input_from_stdin(self) -> None:
        payload = read_damage_calc_input(
            input_path=None,
            use_stdin=True,
            stdin_text=json.dumps(sample_damage_calc_payload()),
        )

        self.assertEqual(payload["defender"]["species"], "Raikou")
        self.assertEqual(payload["move"]["name"], "Double-Edge")

    def test_run_damage_calc_requires_node(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tools").mkdir()
            (root / "tools" / "damage-calc-cli.cjs").write_text("", encoding="utf-8")
            (root / "node_modules" / "@smogon" / "calc").mkdir(parents=True)

            with mock.patch("pokerena.calc.shutil.which", return_value=None):
                with self.assertRaisesRegex(ConfigError, "bootstrap-node-deps"):
                    run_damage_calc(sample_damage_calc_payload(), project_root=root)

    def test_run_damage_calc_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tools").mkdir()
            (root / "tools" / "damage-calc-cli.cjs").write_text("", encoding="utf-8")
            (root / "node_modules" / "@smogon" / "calc").mkdir(parents=True)

            with (
                mock.patch("pokerena.calc.shutil.which", return_value="/usr/bin/node"),
                mock.patch(
                    "pokerena.calc.subprocess.run",
                    side_effect=subprocess.TimeoutExpired(cmd=["node"], timeout=1),
                ),
            ):
                with self.assertRaisesRegex(ConfigError, "timed out"):
                    run_damage_calc(sample_damage_calc_payload(), project_root=root, timeout_seconds=1)

    def test_calc_damage_command_reads_file_input_from_nested_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tools").mkdir()
            (root / "tools" / "damage-calc-cli.cjs").write_text("#!/usr/bin/env node\n", encoding="utf-8")
            (root / "vendor" / "pokemon-showdown").mkdir(parents=True)
            (root / "package.json").write_text("{\"name\":\"pokerena-agent-tools\"}\n", encoding="utf-8")
            nested = root / "nested" / "workdir"
            nested.mkdir(parents=True)
            payload_path = root / "request.json"
            payload_path.write_text(json.dumps(sample_damage_calc_payload()), encoding="utf-8")

            with (
                mock.patch("pokerena.cli.run_damage_calc", return_value={"range": {"min": 1, "max": 2}}),
                mock.patch("sys.stdout", new=StringIO()) as buffer,
            ):
                old_cwd = Path.cwd()
                try:
                    os.chdir(nested)
                    code = main(["calc", "damage", "--input", str(payload_path)])
                finally:
                    os.chdir(old_cwd)

            self.assertEqual(code, 0)
            self.assertIn('"min": 1', buffer.getvalue())

    def test_calc_damage_command_reads_stdin_input(self) -> None:
        payload = json.dumps(sample_damage_calc_payload())

        with (
            mock.patch("pokerena.cli.run_damage_calc", return_value={"range": {"min": 10, "max": 20}}),
            mock.patch("sys.stdin", new=StringIO(payload)),
            mock.patch("sys.stdout", new=StringIO()) as buffer,
        ):
            code = main(["calc", "damage", "--stdin"])

        self.assertEqual(code, 0)
        self.assertIn('"max": 20', buffer.getvalue())

    def test_doctor_reports_missing_calc_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "tools").mkdir()
            (root / "vendor" / "pokemon-showdown").mkdir(parents=True)
            (root / "tools" / "damage-calc-cli.cjs").write_text("#!/usr/bin/env node\n", encoding="utf-8")
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
            (root / "config" / "agents.yaml").write_text("agents: []\n", encoding="utf-8")

            def fake_which(name: str) -> str | None:
                return f"/usr/bin/{name}" if name in {"node", "npm"} else None

            with (
                mock.patch("pokerena.cli.shutil.which", side_effect=fake_which),
                mock.patch("pokerena.cli.node_version", return_value="v22.22.2"),
            ):
                checks = collect_doctor_checks("config/server.local.yaml", "config/agents.yaml", root)

        calc_deps = next(check for check in checks if check.name == "calc-deps")
        calc_smoke = next(check for check in checks if check.name == "calc-smoke")
        self.assertFalse(calc_deps.ok)
        self.assertFalse(calc_smoke.ok)

    def test_doctor_reports_missing_node_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
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
            (root / "config" / "agents.yaml").write_text("agents: []\n", encoding="utf-8")

            with mock.patch("pokerena.cli.shutil.which", return_value=None):
                checks = collect_doctor_checks("config/server.local.yaml", "config/agents.yaml", root)

        calc_smoke = next(check for check in checks if check.name == "calc-smoke")
        self.assertIn("Node.js 22+", calc_smoke.detail)

    def test_doctor_reports_calc_smoke_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "tools").mkdir()
            (root / "node_modules" / "@smogon" / "calc").mkdir(parents=True)
            (root / "vendor" / "pokemon-showdown" / "node_modules").mkdir(parents=True)
            (root / "vendor" / "pokemon-showdown" / "pokemon-showdown").write_text("", encoding="utf-8")
            (root / "tools" / "damage-calc-cli.cjs").write_text("#!/usr/bin/env node\n", encoding="utf-8")
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
            (root / "config" / "agents.yaml").write_text("agents: []\n", encoding="utf-8")

            def fake_which(name: str) -> str | None:
                return f"/usr/bin/{name}" if name in {"node", "npm"} else None

            with (
                mock.patch("pokerena.cli.shutil.which", side_effect=fake_which),
                mock.patch("pokerena.cli.node_version", return_value="v22.22.2"),
                mock.patch("pokerena.cli.run_damage_calc", return_value={"range": {"min": 165, "max": 195}}),
            ):
                checks = collect_doctor_checks("config/server.local.yaml", "config/agents.yaml", root)

        calc_smoke = next(check for check in checks if check.name == "calc-smoke")
        self.assertTrue(calc_smoke.ok)
        self.assertEqual(calc_smoke.detail, "range 165-195")

    def test_detect_project_root_finds_repo_from_nested_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tools").mkdir()
            (root / "tools" / "damage-calc-cli.cjs").write_text("", encoding="utf-8")
            (root / "vendor" / "pokemon-showdown").mkdir(parents=True)
            (root / "package.json").write_text("{\"name\":\"pokerena-agent-tools\"}\n", encoding="utf-8")
            nested = root / "a" / "b" / "c"
            nested.mkdir(parents=True)

            detected = detect_project_root(start=nested)

        self.assertEqual(detected, root.resolve())

    def test_node_cli_rejects_bad_request_shape(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "tools" / "damage-calc-cli.cjs"
        if not script_path.exists() or not shutil.which("node"):
            self.skipTest("node CLI is unavailable")
        if not (repo_root / "node_modules" / "@smogon" / "calc").exists():
            self.skipTest("@smogon/calc is unavailable")

        bad_payload = {
            "schema_version": CALC_REQUEST_SCHEMA_VERSION,
            "generation": 2,
            "attacker": {"species": "Snorlax"},
            "defender": {"species": "Raikou"},
            "move": {},
            "field": {},
        }
        completed = subprocess.run(
            ["node", str(script_path)],
            cwd=repo_root,
            input=json.dumps(bad_payload),
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("move.name", completed.stderr)

    def test_run_damage_calc_returns_expected_gen2_result(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        if not (repo_root / "node_modules" / "@smogon" / "calc").exists():
            self.skipTest("@smogon/calc is unavailable")

        result = run_damage_calc(sample_damage_calc_payload(), project_root=repo_root)

        self.assertEqual(result["schema_version"], "pokerena.damage-result.v1")
        self.assertEqual(result["range"], {"min": 165, "max": 195})
        self.assertEqual(result["range_percent"], {"min": 43.08, "max": 50.91})
        self.assertEqual(result["knockout"]["text"], "guaranteed 3HKO after Leftovers recovery")
