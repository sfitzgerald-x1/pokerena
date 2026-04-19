from __future__ import annotations

import os
from pathlib import Path
import tempfile
import textwrap
import unittest
import random

from pokerena.agent import (
    AgentCancelledError,
    AgentContextCursor,
    BattleSession,
    ShowdownClientAdapter,
    SimStreamAdapter,
    _run_hook_process,
    build_session_from_capture,
    choose_first_legal,
    choose_random_legal,
    find_agent,
    load_capture,
    parse_decision_output,
    render_turn_prompt,
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
            self.assertIn("switch 2", context.legal_action_hints)
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
            self.assertEqual(context.last_error, "[Unavailable choice] Can't move there")
            self.assertTrue(context.signals["request_updated"])

    def test_context_uses_turn_bounded_public_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: test-agent
                        enabled: true
                        provider: claude
                        player_slot: p1
                        format_allowlist: [gen3randombattle]
                        transport: sim-stream
                        launch:
                          command: cat
                          args: []
                          cwd: .
                        hook:
                          history_turn_limit: 2
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            agent = find_agent(load_agents_config(project_root=root), "test-agent")
            session = BattleSession(battle_id="battle-history", player_slot="p1", history_limit=20)
            session.ingest(_public_event("battle-history", ["|turn|1", "|move|p1a: A|Surf|p2a: B"]))
            session.ingest(_public_event("battle-history", ["|turn|2", "|move|p2a: B|Thunderbolt|p1a: A"]))
            session.ingest(_public_event("battle-history", ["|turn|3", "|move|p1a: A|Ice Beam|p2a: B"]))
            session.ingest(
                _request_event(
                    "battle-history",
                    "p1",
                    {
                        "active": [{"moves": [{"move": "Recover", "id": "recover", "disabled": False}]}],
                        "side": {"pokemon": [{"ident": "p1: A", "condition": "50/100", "active": True}]},
                        "rqid": 8,
                    },
                )
            )

            context = session.build_turn_context(
                agent=agent,
                cursor=AgentContextCursor(last_turn_number=2, last_request_sequence=0),
            )

            self.assertNotIn("|turn|1", context.recent_public_events)
            self.assertIn("|turn|2", context.recent_public_events)
            self.assertIn("|turn|3", context.recent_public_events)

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

        mixed_output = parse_decision_output(
            'Correcting my decision first.\n\n{"schema_version":"pokerena.decision.v1","decision":"move 1","notes":"actual json"}',
            "pokerena.decision.v1",
        )
        self.assertEqual(mixed_output.decision, "move 1")
        self.assertEqual(mixed_output.notes, "actual json")

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

    def test_choose_random_legal_covers_common_request_shapes(self) -> None:
        rng = random.Random(7)

        move_choice = choose_random_legal(
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
            },
            rng=rng,
        )
        self.assertIn(move_choice, {"move 1", "move 2"})

        move_or_switch_choice = choose_random_legal(
            {
                "active": [
                    {
                        "moves": [
                            {"move": "Surf", "id": "surf", "disabled": False},
                            {"move": "Ice Beam", "id": "icebeam", "disabled": False},
                        ]
                    }
                ],
                "side": {
                    "pokemon": [
                        {"ident": "p1: Lapras", "condition": "100/100", "active": True},
                        {"ident": "p1: Jolteon", "condition": "100/100", "active": False},
                        {"ident": "p1: Snorlax", "condition": "100/100", "active": False},
                    ]
                },
            },
            rng=rng,
        )
        self.assertIn(move_or_switch_choice, {"move 1", "move 2", "switch 2", "switch 3"})

        switch_choice = choose_random_legal(
            {
                "forceSwitch": [True],
                "side": {
                    "pokemon": [
                        {"ident": "p1: Landorus", "condition": "0 fnt", "active": True},
                        {"ident": "p1: Corviknight", "condition": "100/100", "active": False},
                        {"ident": "p1: Gholdengo", "condition": "100/100", "active": False},
                    ]
                },
            },
            rng=rng,
        )
        self.assertIn(switch_choice, {"switch 2", "switch 3"})

        team_choice = choose_random_legal(
            {
                "teamPreview": True,
                "side": {
                    "pokemon": [
                        {"ident": "p1: A"},
                        {"ident": "p1: B"},
                        {"ident": "p1: C"},
                    ]
                },
            },
            rng=rng,
        )
        self.assertTrue(team_choice.startswith("team "))
        self.assertEqual(sorted(team_choice.replace("team ", "")), ["1", "2", "3"])

    def test_choose_random_legal_avoids_voluntary_switch_when_trapped(self) -> None:
        rng = random.Random(11)
        choice = choose_random_legal(
            {
                "active": [
                    {
                        "moves": [
                            {"move": "Surf", "id": "surf", "disabled": False},
                            {"move": "Ice Beam", "id": "icebeam", "disabled": False},
                        ],
                        "trapped": True,
                    }
                ],
                "side": {
                    "pokemon": [
                        {"ident": "p1: Lapras", "condition": "100/100", "active": True},
                        {"ident": "p1: Jolteon", "condition": "100/100", "active": False},
                    ]
                },
            },
            rng=rng,
        )
        self.assertIn(choice, {"move 1", "move 2"})

    def test_render_turn_prompt_includes_damage_calc_guidance_for_move_turns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: calc-agent
                        enabled: true
                        provider: claude
                        player_slot: p2
                        format_allowlist: [gen3randombattle]
                        transport: showdown-client
                        launch:
                          command: claude
                          args: ["-p"]
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
            agent = load_agents_config(project_root=root)[0]
            session = BattleSession(battle_id="battle-prompt", player_slot="p2", history_limit=20)
            session.ingest(
                _public_event(
                    "battle-prompt",
                    [
                        "|gen|3",
                        "|tier|[Gen 3] Random Battle",
                        "|switch|p1a: Camerupt|Camerupt, L89, F|100/100",
                        "|switch|p2a: Ludicolo|Ludicolo, L84, F|272/272",
                        "|turn|1",
                    ],
                )
            )
            session.ingest(
                _request_event(
                    "battle-prompt",
                    "p2",
                    {
                        "active": [
                            {
                                "moves": [
                                    {
                                        "move": "Hidden Power Grass 70",
                                        "id": "hiddenpower",
                                        "disabled": False,
                                        "category": "Special",
                                        "basePower": 70,
                                    },
                                    {
                                        "move": "Rain Dance",
                                        "id": "raindance",
                                        "disabled": False,
                                        "category": "Status",
                                        "basePower": 0,
                                    },
                                ]
                            }
                        ],
                        "side": {
                            "pokemon": [
                                {
                                    "ident": "p2: Ludicolo",
                                    "details": "Ludicolo, L84, F",
                                    "condition": "272/272",
                                    "active": True,
                                    "item": "leftovers",
                                    "baseAbility": "swiftswim",
                                    "stats": {"atk": 124, "def": 166, "spa": 199, "spd": 216, "spe": 166},
                                }
                            ]
                        },
                        "rqid": 3,
                    },
                )
            )

            context = session.build_turn_context(
                agent=agent,
                cursor=AgentContextCursor(last_turn_number=0, last_request_sequence=0),
            )
            prompt_text = render_turn_prompt(agent, context)

            self.assertIn("DAMAGE CALC WORKFLOW", prompt_text)
            self.assertIn("python3.14 -m pokerena calc damage-batch --stdin", prompt_text)
            self.assertIn("Hidden Power Grass 70", prompt_text)
            self.assertIn("Move 2 · Rain Dance — non-damaging/status move; reason heuristically.", prompt_text)
            self.assertIn("\"generation\": 3", prompt_text)
            self.assertIn("\"species\": \"Camerupt\"", prompt_text)

    def test_render_turn_prompt_skips_damage_calc_guidance_for_switch_turns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: switch-agent
                        enabled: true
                        provider: claude
                        player_slot: p2
                        format_allowlist: [gen3randombattle]
                        transport: showdown-client
                        launch:
                          command: claude
                          args: ["-p"]
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
            agent = load_agents_config(project_root=root)[0]
            session = BattleSession(battle_id="battle-switch", player_slot="p2", history_limit=20)
            session.ingest(_public_event("battle-switch", ["|gen|3", "|tier|[Gen 3] Random Battle", "|turn|4"]))
            session.ingest(
                _request_event(
                    "battle-switch",
                    "p2",
                    {
                        "forceSwitch": [True],
                        "side": {
                            "pokemon": [
                                {"ident": "p2: Ludicolo", "details": "Ludicolo, L84, F", "condition": "0 fnt", "active": True},
                                {"ident": "p2: Glalie", "details": "Glalie, L82, F", "condition": "265/265", "active": False},
                            ]
                        },
                        "rqid": 11,
                    },
                )
            )
            context = session.build_turn_context(
                agent=agent,
                cursor=AgentContextCursor(last_turn_number=3, last_request_sequence=0),
            )

            prompt_text = render_turn_prompt(agent, context)

            self.assertNotIn("DAMAGE CALC WORKFLOW", prompt_text)

    def test_render_turn_prompt_mentions_voluntary_switching_on_move_turns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: switch-aware-agent
                        enabled: true
                        provider: claude
                        player_slot: p1
                        format_allowlist: [gen3randombattle]
                        transport: showdown-client
                        launch:
                          command: claude
                          args: ["-p"]
                          cwd: .
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            agent = load_agents_config(project_root=root)[0]
            session = BattleSession(battle_id="battle-voluntary-switch", player_slot="p1", history_limit=20)
            session.ingest(_public_event("battle-voluntary-switch", ["|gen|3", "|tier|[Gen 3] Random Battle", "|turn|6"]))
            session.ingest(
                _request_event(
                    "battle-voluntary-switch",
                    "p1",
                    {
                        "active": [
                            {
                                "moves": [
                                    {"move": "Surf", "id": "surf", "disabled": False},
                                    {"move": "Rain Dance", "id": "raindance", "disabled": False, "category": "Status"},
                                ]
                            }
                        ],
                        "side": {
                            "pokemon": [
                                {"ident": "p1: Lapras", "details": "Lapras, L76, F", "condition": "100/100", "active": True},
                                {"ident": "p1: Jolteon", "details": "Jolteon, L78, M", "condition": "100/100", "active": False},
                            ]
                        },
                        "rqid": 14,
                    },
                )
            )
            context = session.build_turn_context(
                agent=agent,
                cursor=AgentContextCursor(last_turn_number=5, last_request_sequence=0),
            )

            prompt_text = render_turn_prompt(agent, context)

            self.assertIn("VOLUNTARY SWITCHING", prompt_text)
            self.assertIn("switch 2", prompt_text)

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

    def test_sim_stream_adapter_ignores_launcher_preamble(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        showdown_path = repo_root / "vendor" / "pokemon-showdown"
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
        )

        events = adapter._parse_chunk(
            "config.js does not exist. Creating one with default settings...\nupdate\n|turn|1"
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "public_update")
        self.assertEqual(events[0].payload["lines"], ["|turn|1"])

    def test_showdown_client_adapter_accepts_supported_direct_challenge(self) -> None:
        adapter = ShowdownClientAdapter(server_config=_server_config(), agent=_load_showdown_agent())
        fake_connection = _FakeConnection()
        adapter.connection = fake_connection
        adapter.authenticated = True

        events = adapter._consume_message(
            '|updatechallenges|{"challengesFrom":{"human":"gen3randombattle"}}'
        )

        self.assertEqual(events, [])
        self.assertEqual(fake_connection.sent, ["|/utm null", "|/accept human"])
        self.assertEqual(adapter.pending_challenger, "human")
        self.assertEqual(adapter.pending_format, "gen3randombattle")

    def test_showdown_client_adapter_rejects_unsupported_direct_challenge(self) -> None:
        adapter = ShowdownClientAdapter(server_config=_server_config(), agent=_load_showdown_agent())
        fake_connection = _FakeConnection()
        adapter.connection = fake_connection
        adapter.authenticated = True

        events = adapter._consume_message(
            '|updatechallenges|{"challengesFrom":{"human":"gen4randombattle"}}'
        )

        self.assertEqual(events, [])
        self.assertEqual(
            fake_connection.sent,
            [
                "|/pm human, I only accept these formats: gen3randombattle.",
                "|/reject human",
            ],
        )

    def test_showdown_client_adapter_accepts_pm_challenge_messages(self) -> None:
        adapter = ShowdownClientAdapter(server_config=_server_config(), agent=_load_showdown_agent())
        fake_connection = _FakeConnection()
        adapter.connection = fake_connection
        adapter.authenticated = True

        events = adapter._consume_message(
            "|pm| human123| ClaudeLocalBot|/challenge gen3randombattle|gen3randombattle|||"
        )

        self.assertEqual(events, [])
        self.assertEqual(fake_connection.sent, ["|/utm null", "|/accept human123"])

    def test_showdown_client_adapter_parses_battle_room_and_submits_choice(self) -> None:
        adapter = ShowdownClientAdapter(server_config=_server_config(), agent=_load_showdown_agent())
        fake_connection = _FakeConnection()
        adapter.connection = fake_connection
        adapter.authenticated = True
        adapter.pending_challenger = "human"
        adapter.pending_format = "gen3randombattle"

        events = adapter._consume_message(
            textwrap.dedent(
                """
                >battle-gen3randombattle-1
                |init|battle
                |title|ClaudeLocalBot vs. human
                |tier|[Gen 3] Random Battle
                |turn|1
                |request|{"rqid":7,"active":[{"moves":[{"move":"Surf","id":"surf","disabled":false}]}],"side":{"pokemon":[{"ident":"p1: Lapras","condition":"100/100","active":true}]}}
                """
            ).strip()
        )

        self.assertEqual(events[0].event_type, "battle_started")
        self.assertEqual(events[0].battle_id, "battle-gen3randombattle-1")
        self.assertEqual(events[1].event_type, "public_update")
        self.assertIn("|tier|[Gen 3] Random Battle", events[1].payload["lines"])
        self.assertEqual(events[2].event_type, "request_received")
        self.assertEqual(events[2].payload["rqid"], 7)

        adapter.submit_decision(player_slot="p1", choice="move 1", rqid="7")
        self.assertEqual(fake_connection.sent, ["battle-gen3randombattle-1|/choose move 1|7"])

    def test_showdown_client_adapter_ignores_malformed_updateuser_frames(self) -> None:
        adapter = ShowdownClientAdapter(server_config=_server_config(), agent=_load_showdown_agent())
        fake_connection = _FakeConnection()
        adapter.connection = fake_connection

        events = adapter._consume_message("|updateuser|too-short")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "client_notice")
        self.assertIn("malformed Showdown |updateuser|", events[0].payload["message"])
        self.assertFalse(adapter.authenticated)

        duplicate_events = adapter._consume_message("|updateuser|too-short")
        self.assertEqual(duplicate_events, [])

    def test_showdown_client_adapter_emits_notice_for_malformed_updatechallenges(self) -> None:
        adapter = ShowdownClientAdapter(server_config=_server_config(), agent=_load_showdown_agent())
        fake_connection = _FakeConnection()
        adapter.connection = fake_connection

        events = adapter._consume_message("|updatechallenges|not-json")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "client_notice")
        self.assertIn("malformed Showdown |updatechallenges|", events[0].payload["message"])

    def test_showdown_client_adapter_forfeits_unexpected_second_battle_room(self) -> None:
        adapter = ShowdownClientAdapter(server_config=_server_config(), agent=_load_showdown_agent())
        fake_connection = _FakeConnection()
        adapter.connection = fake_connection
        adapter.authenticated = True
        adapter.current_battle_id = "battle-gen3randombattle-1"

        events = adapter._consume_message(
            textwrap.dedent(
                """
                >battle-gen3randombattle-2
                |init|battle
                |title|ClaudeLocalBot vs. human-two
                """
            ).strip()
        )

        self.assertEqual(fake_connection.sent, ["battle-gen3randombattle-2|/forfeit"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "client_notice")
        self.assertIn("only support one active battle", events[0].payload["message"])

    def test_hook_process_can_be_cancelled(self) -> None:
        with self.assertRaises(AgentCancelledError) as error:
            _run_hook_process(
                command=["python3.14", "-c", "import time; time.sleep(30)"],
                cwd=Path.cwd(),
                env=os.environ.copy(),
                prompt_text="prompt",
                timeout_seconds=30,
                expected_schema="pokerena.decision.v1",
                claude_streaming=False,
                trace_sink=None,
                cancel_check=lambda: "Battle manually stopped from Battle Sessions.",
            )

        self.assertIn("manually stopped", str(error.exception))


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


def _server_config() -> ServerConfig:
    repo_root = Path(__file__).resolve().parents[1]
    return ServerConfig(
        project_root=repo_root,
        config_path=repo_root / "config" / "server.local.example.yaml",
        showdown_path=repo_root / "vendor" / "pokemon-showdown",
        bind_address="127.0.0.1",
        port=8000,
        server_id="pokerena-local",
        public_origin="http://localhost:8000",
        no_security=True,
        data_dir=repo_root / ".runtime" / "showdown" / "data",
        log_dir=repo_root / ".runtime" / "showdown" / "logs",
        runtime_dir=repo_root / ".runtime" / "showdown",
    )


def _load_showdown_agent():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "config").mkdir()
        (root / "config" / "agents.yaml").write_text(
            textwrap.dedent(
                """
                agents:
                  - id: showdown-agent
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
                      challenge_policy: accept-direct-challenges
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return find_agent(load_agents_config(project_root=root), "showdown-agent")


class _FakeConnection:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, payload: str) -> None:
        self.sent.append(payload)

    def close(self) -> None:
        return
