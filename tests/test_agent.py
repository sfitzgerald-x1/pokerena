from __future__ import annotations

from pathlib import Path
import tempfile
import textwrap
import unittest

from pokerena.agent import (
    AgentContextCursor,
    BattleSession,
    SimStreamAdapter,
    build_session_from_capture,
    choose_first_legal,
    find_agent,
    load_capture,
    parse_decision_output,
    save_capture,
)
from pokerena.config import ServerConfig, load_agents_config


class BattleAgentRuntimeTest(unittest.TestCase):
    def test_session_builds_context_from_request_events(self) -> None:
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
                        hook:
                          type: subprocess_stdio
                          context_format: pokerena.turn-context.v1
                          decision_format: pokerena.decision.v1
                          prompt_style: showdown-turn-v1
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            agent = find_agent(load_agents_config(project_root=root), "test-agent")
            session = BattleSession(battle_id="battle-1", player_slot="p1", history_limit=20)
            session.ingest(
                _public_event(
                    "battle-1",
                    [
                        "|tier|[Gen 9] Random Battle",
                        "|turn|3",
                    ],
                )
            )
            session.ingest(
                _request_event(
                    "battle-1",
                    "p1",
                    {
                        "active": [
                            {
                                "moves": [
                                    {"move": "Thunderbolt", "id": "thunderbolt", "disabled": False},
                                    {"move": "Surf", "id": "surf", "disabled": True},
                                ]
                            }
                        ],
                        "side": {
                            "pokemon": [
                                {"ident": "p1: Pikachu", "condition": "100/100", "active": True},
                                {"ident": "p1: Charizard", "condition": "100/100", "active": False},
                            ]
                        },
                    },
                )
            )

            context = session.build_turn_context(
                agent=agent,
                cursor=AgentContextCursor(last_turn_number=None, last_request_sequence=0),
            )

            self.assertEqual(context.rqid, "sim-1")
            self.assertEqual(context.request_kind, "move")
            self.assertTrue(context.signals["turn_started"])
            self.assertTrue(context.signals["request_updated"])
            self.assertIn("move 1", context.legal_action_hints)
            self.assertEqual(context.turn_number, 3)

    def test_invalid_choice_update_reuses_synthetic_request_id(self) -> None:
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
            agent = find_agent(load_agents_config(project_root=root), "test-agent")
            session = BattleSession(battle_id="battle-2", player_slot="p1", history_limit=20)
            session.ingest(_public_event("battle-2", ["|turn|5"]))
            initial_request = {
                "active": [{"moves": [{"move": "Recover", "id": "recover", "disabled": False}]}],
                "side": {"pokemon": [{"ident": "p1: Mew", "condition": "100/100", "active": True}]},
            }
            session.ingest(_request_event("battle-2", "p1", initial_request))
            first_rqid = session.current_rqid

            session.ingest(
                _rejection_event("battle-2", "p1", "[Unavailable choice] Can't move there")
            )
            updated_request = dict(initial_request)
            updated_request["update"] = True
            session.ingest(_request_event("battle-2", "p1", updated_request))

            context = session.build_turn_context(
                agent=agent,
                cursor=AgentContextCursor(last_turn_number=4, last_request_sequence=1),
            )

            self.assertEqual(context.rqid, first_rqid)
            self.assertEqual(context.decision_attempt, 2)
            self.assertEqual(context.last_error, None)
            self.assertTrue(context.signals["request_updated"])

    def test_capture_replay_reconstructs_session_state(self) -> None:
        session = BattleSession(battle_id="battle-3", player_slot="p1", history_limit=20)
        session.ingest(_public_event("battle-3", ["|tier|[Gen 9] Random Battle", "|turn|1"]))
        session.ingest(
            _request_event(
                "battle-3",
                "p1",
                {
                    "active": [{"moves": [{"move": "Moonblast", "id": "moonblast", "disabled": False}]}],
                    "side": {"pokemon": [{"ident": "p1: Flutter Mane", "condition": "100/100", "active": True}]},
                },
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            capture_path = Path(temp_dir) / "capture.json"
            save_capture(capture_path, session.to_capture())
            replay = load_capture(capture_path)
            rebuilt = build_session_from_capture(capture=replay, player_slot="p1", history_limit=20)

            self.assertEqual(rebuilt.battle_id, "battle-3")
            self.assertEqual(rebuilt.current_rqid, "sim-1")
            self.assertEqual(rebuilt.turn_number, 1)
            self.assertEqual(rebuilt.format_name, "[Gen 9] Random Battle")

    def test_multi_active_hints_do_not_flatten_move_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: doubles-agent
                        enabled: true
                        provider: codex
                        player_slot: p1
                        format_allowlist: [gen4doublescustomgame]
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
            agent = find_agent(load_agents_config(project_root=root), "doubles-agent")
            session = BattleSession(battle_id="battle-doubles", player_slot="p1", history_limit=20)
            session.ingest(_public_event("battle-doubles", ["|turn|2"]))
            session.ingest(
                _request_event(
                    "battle-doubles",
                    "p1",
                    {
                        "active": [
                            {"moves": [{"move": "Surf", "id": "surf", "disabled": False}]},
                            {"moves": [{"move": "Protect", "id": "protect", "disabled": False}]},
                        ],
                        "side": {
                            "pokemon": [
                                {"ident": "p1a: Lapras", "condition": "100/100", "active": True},
                                {"ident": "p1b: Snorlax", "condition": "100/100", "active": True},
                            ]
                        },
                    },
                )
            )

            context = session.build_turn_context(
                agent=agent,
                cursor=AgentContextCursor(last_turn_number=1, last_request_sequence=0),
            )

            self.assertEqual(context.legal_action_hints, ["wait"])

    def test_parse_decision_output_accepts_json_and_plain_text(self) -> None:
        json_decision = parse_decision_output(
            '{"schema_version":"pokerena.decision.v1","decision":"move 1","notes":"safe"}',
            "pokerena.decision.v1",
        )
        self.assertEqual(json_decision.decision, "move 1")

        plain_decision = parse_decision_output("switch 2\n", "pokerena.decision.v1")
        self.assertEqual(plain_decision.decision, "switch 2")

    def test_choose_first_legal_covers_common_request_shapes(self) -> None:
        move_choice = choose_first_legal(
            {
                "active": [{"moves": [{"move": "Surf", "id": "surf", "disabled": False}]}],
                "side": {"pokemon": [{"ident": "p1: Lapras", "condition": "100/100", "active": True}]},
            }
        )
        self.assertEqual(move_choice, "move 1")

        switch_choice = choose_first_legal(
            {
                "forceSwitch": [True],
                "side": {
                    "pokemon": [
                        {"ident": "p1: Landorus", "condition": "0 fnt", "active": True},
                        {"ident": "p1: Corviknight", "condition": "100/100", "active": False},
                    ]
                },
            }
        )
        self.assertEqual(switch_choice, "switch 2")

        team_choice = choose_first_legal(
            {
                "teamPreview": True,
                "side": {
                    "pokemon": [
                        {"ident": "p1: A"},
                        {"ident": "p1: B"},
                        {"ident": "p1: C"},
                    ]
                },
            }
        )
        self.assertEqual(team_choice, "team 123")

    def test_sim_stream_adapter_round_trips_real_requests(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        showdown_path = repo_root / "vendor" / "pokemon-showdown"
        if not (showdown_path / "pokemon-showdown").exists():
            self.skipTest("pokemon-showdown launcher is unavailable")

        server_config = ServerConfig(
            project_root=repo_root,
            config_path=repo_root / "config" / "server.local.example.yaml",
            showdown_path=showdown_path,
            bind_address="127.0.0.1",
            port=8000,
            server_id="pokerena-local",
            public_origin="http://localhost:8000",
            no_security=True,
            data_dir=repo_root / ".runtime" / "showdown" / "data",
            log_dir=repo_root / ".runtime" / "showdown" / "logs",
            runtime_dir=repo_root / ".runtime" / "showdown",
        )
        adapter = SimStreamAdapter(
            server_config=server_config,
            format_id="gen9randombattle",
            battle_id="sim-test",
            player_names={"p1": "Alpha", "p2": "Beta"},
            seed=[1, 2, 3, 4],
        )

        saw_public = False
        p1_request = None
        p2_request = None
        follow_up_public = False

        try:
            adapter.connect()
            for _ in range(10):
                event = adapter.next_event()
                if event.event_type == "public_update":
                    saw_public = True
                if event.event_type == "request_received" and event.player_slot == "p1":
                    p1_request = event.payload
                if event.event_type == "request_received" and event.player_slot == "p2":
                    p2_request = event.payload
                if saw_public and p1_request is not None and p2_request is not None:
                    break

            self.assertTrue(saw_public)
            self.assertIsNotNone(p1_request)
            self.assertIsNotNone(p2_request)

            adapter.submit_decision(player_slot="p1", choice=choose_first_legal(p1_request), rqid=None)
            adapter.submit_decision(player_slot="p2", choice=choose_first_legal(p2_request), rqid=None)

            for _ in range(12):
                event = adapter.next_event()
                if event.event_type == "public_update":
                    lines = event.payload.get("lines", [])
                    if any("|move|" in line for line in lines) or any("|turn|2" == line for line in lines):
                        follow_up_public = True
                        break

            self.assertTrue(follow_up_public)
        finally:
            adapter.close()


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
