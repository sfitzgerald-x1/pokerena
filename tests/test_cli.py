import os
from io import StringIO
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest import mock

from pokerena.agent import AgentContextCursor, AgentTimeoutError, BattleSession, DecisionPolicy, save_capture
from pokerena.cli import (
    _decide_or_fallback,
    _selected_action_label,
    _update_submission_validation,
    collect_doctor_checks,
    main,
)
from pokerena.config import ConfigError, load_agents_config
from pokerena.transcript import load_battle_transcript


class CLITest(unittest.TestCase):
    def test_selected_action_label_formats_move_and_switch_choices(self) -> None:
        move_request = {
            "active": [{"moves": [{"move": "Thunderbolt", "id": "thunderbolt", "disabled": False}]}],
            "side": {"pokemon": [{"ident": "p1: Pikachu", "condition": "100/100", "active": True}]},
        }
        switch_request = {
            "forceSwitch": [True],
            "side": {
                "pokemon": [
                    {"ident": "p1: Pikachu", "details": "Pikachu, L50", "condition": "0 fnt", "active": True},
                    {"ident": "p1: Snorlax", "details": "Snorlax, L50", "condition": "100/100"},
                ]
            },
        }

        self.assertEqual(_selected_action_label("move 1", move_request), 'Used "Thunderbolt"')
        self.assertEqual(_selected_action_label("switch 2", switch_request), 'Switch to "Snorlax"')

    def test_selected_action_label_returns_none_for_unmappable_choices(self) -> None:
        request = {
            "active": [{"moves": [{"move": "Surf", "id": "surf", "disabled": False}]}],
            "side": {"pokemon": [{"ident": "p1: Lapras", "condition": "100/100", "active": True}]},
        }

        self.assertIsNone(_selected_action_label("move 2", request))
        self.assertIsNone(_selected_action_label("team 123456", request))
        self.assertIsNone(_selected_action_label("move 1", {"active": [{}, {}]}))

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
            (root / "config" / "agents.yaml").write_text("agents: []\n", encoding="utf-8")

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
            (root / "config" / "agents.yaml").write_text("agents: []\n", encoding="utf-8")

            with mock.patch("sys.stdout", new=StringIO()) as buffer:
                old_cwd = Path.cwd()
                try:
                    os.chdir(root)
                    code = main(["server", "up", "--config", "config/server.local.yaml", "--dry-run"])
                finally:
                    os.chdir(old_cwd)

            self.assertEqual(code, 0)
            self.assertIn("node", buffer.getvalue())

    def test_server_up_starts_callable_agents_and_shuts_them_down(self) -> None:
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
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: callable-agent
                        enabled: true
                        provider: claude
                        player_slot: p1
                        format_allowlist: [gen3randombattle]
                        transport: showdown-client
                        launch:
                          command: cat
                          args: []
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
            (root / "vendor" / "pokemon-showdown" / "pokemon-showdown").write_text("", encoding="utf-8")
            (root / "vendor" / "pokemon-showdown" / "node_modules").mkdir()

            server_process = _FakeProcess()
            viewer_process = _FakeProcess()
            agent_process = _FakeProcess()
            with (
                mock.patch("pokerena.cli.shutil.which", side_effect=lambda name: f"/usr/bin/{name}"),
                mock.patch("pokerena.cli.node_version", return_value="v22.0.0"),
                mock.patch("pokerena.cli._wait_for_server_ready"),
                mock.patch("pokerena.cli._wait_for_http_ready"),
                mock.patch("pokerena.cli.time.sleep", side_effect=KeyboardInterrupt),
                mock.patch("pokerena.cli.subprocess.Popen", side_effect=[server_process, viewer_process, agent_process]) as popen,
                mock.patch("sys.stdout", new=StringIO()),
            ):
                old_cwd = Path.cwd()
                try:
                    os.chdir(root)
                    code = main(
                        [
                            "server",
                            "up",
                            "--config",
                            "config/server.local.yaml",
                            "--agents-config",
                            "config/agents.yaml",
                        ]
                    )
                finally:
                    os.chdir(old_cwd)

            self.assertEqual(code, 130)
            self.assertEqual(popen.call_count, 3)
            self.assertTrue(server_process.terminated)
            self.assertTrue(viewer_process.terminated)
            self.assertTrue(agent_process.terminated)

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
            transcript_path = root / ".runtime" / "agents" / agent.agent_id / "battle-retry" / "transcript.json"

            with mock.patch("pokerena.cli.invoke_agent") as invoke_agent:
                decision = _decide_or_fallback(
                    agent=agent,
                    session=session,
                    context=context,
                    runtime_root=root / ".runtime",
                    capture_path=root / "capture.json",
                    transcript_path=transcript_path,
                    policy=DecisionPolicy(max_invalid_retries=3, decision_timeout_seconds=10, history_limit=60),
                )

            self.assertEqual(decision, "move 1")
            invoke_agent.assert_not_called()
            transcript = load_battle_transcript(root / ".runtime", agent.agent_id, "battle-retry")
            self.assertIsNotNone(transcript)
            self.assertTrue(transcript["entries"][0]["fallback_used"])
            self.assertEqual(transcript["entries"][0]["fallback_reason"], "max-invalid-retries")
            self.assertEqual(transcript["entries"][0]["submission_state"], "pending")
            self.assertEqual(transcript["entries"][0]["selected_action"], "move 1")
            self.assertEqual(transcript["entries"][0]["selected_action_label"], 'Used "Thunderbolt"')
            self.assertEqual(transcript["entries"][0]["selected_action_source"], "fallback-first-legal")

    def test_decide_or_fallback_appends_transcript_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: transcript-agent
                        enabled: true
                        provider: claude
                        player_slot: p1
                        format_allowlist: [gen3randombattle]
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
            session = BattleSession(battle_id="battle-transcript", player_slot="p1")
            session.ingest(_public_event("battle-transcript", ["|turn|4"]))
            session.ingest(
                _request_event(
                    "battle-transcript",
                    "p1",
                    {
                        "active": [{"moves": [{"move": "Thunderbolt", "id": "thunderbolt", "disabled": False}]}],
                        "side": {"pokemon": [{"ident": "p1: Pikachu", "condition": "100/100", "active": True}]},
                        "rqid": 9,
                    },
                )
            )
            context = session.build_turn_context(
                agent=agent,
                cursor=AgentContextCursor(last_turn_number=3, last_request_sequence=0),
            )
            transcript_path = root / ".runtime" / "agents" / agent.agent_id / "battle-transcript" / "transcript.json"

            with mock.patch(
                "pokerena.cli.invoke_agent",
                return_value=(
                    mock.Mock(
                        decision="move 1",
                        notes="safe move",
                        raw_output='{"decision":"move 1"}',
                    ),
                    mock.Mock(),
                ),
            ):
                first = _decide_or_fallback(
                    agent=agent,
                    session=session,
                    context=context,
                    runtime_root=root / ".runtime",
                    capture_path=root / "capture.json",
                    transcript_path=transcript_path,
                    policy=DecisionPolicy(max_invalid_retries=3, decision_timeout_seconds=10, history_limit=60),
                )

            self.assertEqual(first, "move 1")

            session.ingest(_rejection_event("battle-transcript", "p1", "[Unavailable choice] Can't move there"))
            session.ingest(
                _request_event(
                    "battle-transcript",
                    "p1",
                    {
                        "active": [{"moves": [{"move": "Thunderbolt", "id": "thunderbolt", "disabled": False}]}],
                        "side": {"pokemon": [{"ident": "p1: Pikachu", "condition": "100/100", "active": True}]},
                        "rqid": 9,
                        "update": True,
                    },
                )
            )
            retry_context = session.build_turn_context(
                agent=agent,
                cursor=AgentContextCursor(last_turn_number=4, last_request_sequence=1),
            )
            with mock.patch("pokerena.cli.invoke_agent", side_effect=ConfigError("boom")):
                second = _decide_or_fallback(
                    agent=agent,
                    session=session,
                    context=retry_context,
                    runtime_root=root / ".runtime",
                    capture_path=root / "capture.json",
                    transcript_path=transcript_path,
                    policy=DecisionPolicy(max_invalid_retries=3, decision_timeout_seconds=10, history_limit=60),
                )

            self.assertEqual(second, "move 1")
            transcript = load_battle_transcript(root / ".runtime", agent.agent_id, "battle-transcript")
            self.assertIsNotNone(transcript)
            self.assertEqual(len(transcript["entries"]), 2)
            self.assertFalse(transcript["entries"][0]["fallback_used"])
            self.assertTrue(transcript["entries"][1]["fallback_used"])
            self.assertEqual(transcript["entries"][0]["submission_state"], "pending")
            self.assertEqual(transcript["entries"][1]["submission_state"], "pending")
            self.assertEqual(transcript["entries"][0]["selected_action"], "move 1")
            self.assertEqual(transcript["entries"][0]["selected_action_label"], 'Used "Thunderbolt"')
            self.assertEqual(transcript["entries"][0]["selected_action_source"], "agent")
            self.assertEqual(transcript["entries"][1]["selected_action"], "move 1")
            self.assertEqual(transcript["entries"][1]["selected_action_label"], 'Used "Thunderbolt"')
            self.assertEqual(transcript["entries"][1]["selected_action_source"], "fallback-first-legal")

    def test_timeout_uses_random_legal_fallback_and_records_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: timeout-agent
                        enabled: true
                        provider: claude
                        player_slot: p1
                        format_allowlist: [gen3randombattle]
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
            session = BattleSession(battle_id="battle-timeout", player_slot="p1")
            session.ingest(_public_event("battle-timeout", ["|turn|1"]))
            session.ingest(
                _request_event(
                    "battle-timeout",
                    "p1",
                    {
                        "active": [
                            {
                                "moves": [
                                    {"move": "Surf", "id": "surf", "disabled": False},
                                    {"move": "Ice Beam", "id": "icebeam", "disabled": False},
                                ]
                            }
                        ],
                        "side": {"pokemon": [{"ident": "p1: Lapras", "condition": "100/100", "active": True}]},
                        "rqid": 7,
                    },
                )
            )
            context = session.build_turn_context(
                agent=agent,
                cursor=AgentContextCursor(last_turn_number=0, last_request_sequence=0),
            )
            transcript_path = root / ".runtime" / "agents" / agent.agent_id / "battle-timeout" / "transcript.json"

            with mock.patch("pokerena.cli.invoke_agent", side_effect=AgentTimeoutError("timed out")):
                with mock.patch("pokerena.cli.choose_random_legal", return_value="move 2"):
                    decision = _decide_or_fallback(
                        agent=agent,
                        session=session,
                        context=context,
                        runtime_root=root / ".runtime",
                        capture_path=root / "capture.json",
                        transcript_path=transcript_path,
                        policy=DecisionPolicy(max_invalid_retries=3, decision_timeout_seconds=120, history_limit=60),
                    )

            self.assertEqual(decision, "move 2")
            transcript = load_battle_transcript(root / ".runtime", agent.agent_id, "battle-timeout")
            self.assertIsNotNone(transcript)
            entry = transcript["entries"][0]
            self.assertEqual(entry["fallback_reason"], "timeout")
            self.assertEqual(entry["selected_action"], "move 2")
            self.assertEqual(entry["selected_action_label"], 'Used "Ice Beam"')
            self.assertEqual(entry["selected_action_source"], "timeout-random")
            self.assertEqual(entry["turn_state"], "timed_out")
            self.assertEqual(entry["timeout_seconds"], 120)
            self.assertTrue(entry["trace_events"])
            self.assertIn("Timed out after 120s", entry["trace_events"][-1]["message"])

    def test_submission_validation_marks_rejected_and_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: validation-agent
                        enabled: true
                        provider: claude
                        player_slot: p1
                        format_allowlist: [gen3randombattle]
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
            transcript_path = root / ".runtime" / "agents" / agent.agent_id / "battle-validate" / "transcript.json"
            session = BattleSession(battle_id="battle-validate", player_slot="p1")
            session.ingest(_public_event("battle-validate", ["|turn|1"]))
            session.ingest(
                _request_event(
                    "battle-validate",
                    "p1",
                    {
                        "active": [{"moves": [{"move": "Surf", "id": "surf", "disabled": False}]}],
                        "side": {"pokemon": [{"ident": "p1: Lapras", "condition": "100/100", "active": True}]},
                        "rqid": 7,
                    },
                )
            )
            context = session.build_turn_context(
                agent=agent,
                cursor=AgentContextCursor(last_turn_number=0, last_request_sequence=0),
            )
            with mock.patch(
                "pokerena.cli.invoke_agent",
                return_value=(
                    mock.Mock(
                        decision="move 1",
                        notes="safe move",
                        raw_output='{"decision":"move 1"}',
                    ),
                    mock.Mock(),
                ),
            ):
                _decide_or_fallback(
                    agent=agent,
                    session=session,
                    context=context,
                    runtime_root=root / ".runtime",
                    capture_path=root / "capture.json",
                    transcript_path=transcript_path,
                    policy=DecisionPolicy(max_invalid_retries=3, decision_timeout_seconds=10, history_limit=60),
                )

            pending = _update_submission_validation(
                event=_rejection_event("battle-validate", "p1", "[Invalid choice] Bad command"),
                session=session,
                transcript_path=transcript_path,
                pending_request_sequence=context.request_sequence,
            )
            self.assertIsNone(pending)
            transcript = load_battle_transcript(root / ".runtime", agent.agent_id, "battle-validate")
            self.assertEqual(transcript["entries"][0]["submission_state"], "rejected")
            self.assertEqual(transcript["entries"][0]["turn_state"], "rejected")

            session.ingest(
                _request_event(
                    "battle-validate",
                    "p1",
                    {
                        "active": [{"moves": [{"move": "Surf", "id": "surf", "disabled": False}]}],
                        "side": {"pokemon": [{"ident": "p1: Lapras", "condition": "100/100", "active": True}]},
                        "rqid": 9,
                    },
                )
            )
            retry_context = session.build_turn_context(
                agent=agent,
                cursor=AgentContextCursor(last_turn_number=1, last_request_sequence=1),
            )
            with mock.patch(
                "pokerena.cli.invoke_agent",
                return_value=(
                    mock.Mock(
                        decision="move 1",
                        notes="retry",
                        raw_output='{"decision":"move 1"}',
                    ),
                    mock.Mock(),
                ),
            ):
                _decide_or_fallback(
                    agent=agent,
                    session=session,
                    context=retry_context,
                    runtime_root=root / ".runtime",
                    capture_path=root / "capture.json",
                    transcript_path=transcript_path,
                    policy=DecisionPolicy(max_invalid_retries=3, decision_timeout_seconds=10, history_limit=60),
                )

            pending = _update_submission_validation(
                event=_public_event("battle-validate", ["|move|p1a: Lapras|Surf|p2a: Target"]),
                session=session,
                transcript_path=transcript_path,
                pending_request_sequence=retry_context.request_sequence,
            )
            self.assertIsNone(pending)
            transcript = load_battle_transcript(root / ".runtime", agent.agent_id, "battle-validate")
            self.assertEqual(transcript["entries"][1]["submission_state"], "accepted")
            self.assertEqual(transcript["entries"][1]["turn_state"], "accepted")

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


class _FakeProcess:
    def __init__(self) -> None:
        self.stdout = StringIO("")
        self.stderr = StringIO("")
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode or 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
