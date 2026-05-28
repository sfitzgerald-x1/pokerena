from __future__ import annotations

import random
from pathlib import Path
import unittest
from unittest import mock

from pokerena.config import ConfigError
from pokerena.custom_bot import (
    _choose_weighted,
    _revealed_opponent_moves,
    _scored_action,
    _sleep_clause_active,
    _status_base_value,
    build_custom_bot_plan,
)


class CustomBotTest(unittest.TestCase):
    def test_damage_scores_include_accuracy_and_square_law_weighting(self) -> None:
        context = _context(
            [
                {"move": "Thunder", "id": "thunder", "disabled": False},
                {"move": "Swift", "id": "swift", "disabled": False},
            ]
        )
        metadata = {
            "Thunder": _metadata("Thunder", accuracy=70),
            "Swift": _metadata("Swift", accuracy=100),
        }
        ranges = {"Thunder": (70, 70), "Swift": (70, 70)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(1))

        by_choice = {action.choice: action for action in plan.actions}
        self.assertLess(by_choice["move 1"].score, by_choice["move 2"].score)
        self.assertAlmostEqual(by_choice["move 2"].weight, by_choice["move 2"].score ** 2)
        self.assertIn("custom-bot gen1randombattle weighted scores", plan.notes)

    def test_status_move_scores_high_against_non_statused_target(self) -> None:
        context = _context(
            [
                {"move": "Tackle", "id": "tackle", "disabled": False},
                {"move": "Sleep Powder", "id": "sleeppowder", "disabled": False},
            ]
        )
        metadata = {
            "Tackle": _metadata("Tackle", base_power=40),
            "Sleep Powder": _metadata(
                "Sleep Powder",
                category="Status",
                base_power=0,
                accuracy=75,
                status="slp",
            ),
        }
        ranges = {"Tackle": (20, 25)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(2))

        by_choice = {action.choice: action for action in plan.actions}
        self.assertGreater(by_choice["move 2"].score, by_choice["move 1"].score)

    def test_sleep_moves_score_zero_when_sleep_clause_is_active(self) -> None:
        context = _context(
            [{"move": "Sleep Powder", "id": "sleeppowder", "disabled": False}],
            recent_public_events=[
                "|gen|1",
                "|switch|p1a: Venusaur|Venusaur, L80|100/100",
                "|switch|p2a: Starmie|Starmie, L80|100/100",
                "|turn|4",
            ],
        )
        capture = {
            "events": [
                {"payload": {"lines": ["|gen|1", "|switch|p1a: Venusaur|Venusaur, L80|100/100"]}},
                {
                    "payload": {
                        "lines": [
                            "|switch|p2a: Alakazam|Alakazam, L80|100/100",
                            "|move|p1a: Venusaur|Sleep Powder|p2a: Alakazam",
                            "|-status|p2a: Alakazam|slp",
                            "|switch|p2a: Starmie|Starmie, L80|100/100",
                        ]
                    }
                },
            ]
        }
        metadata = {
            "Sleep Powder": _metadata(
                "Sleep Powder",
                category="Status",
                base_power=0,
                accuracy=75,
                status="slp",
            )
        }

        with _patched_calc(metadata, {}):
            plan = build_custom_bot_plan(
                context,
                capture_payload=capture,
                project_root=Path.cwd(),
                rng=random.Random(3),
            )

        self.assertEqual(plan.actions, [])
        self.assertEqual(plan.fallback_reason, "all heuristic scores were zero")

    def test_rest_sleep_does_not_activate_sleep_clause(self) -> None:
        context = _context(
            [{"move": "Sleep Powder", "id": "sleeppowder", "disabled": False}],
        )
        public_lines = [
            "|gen|1",
            "|switch|p1a: Venusaur|Venusaur, L80|100/100",
            "|switch|p2a: Starmie|Starmie, L80|100/100",
            "|move|p2a: Starmie|Rest|p2a: Starmie",
            "|-status|p2a: Starmie|slp",
        ]

        self.assertFalse(_sleep_clause_active(context, public_lines))

    def test_status_move_scores_zero_when_target_already_statused(self) -> None:
        context = _context(
            [{"move": "Thunder Wave", "id": "thunderwave", "disabled": False}],
            recent_public_events=[
                "|gen|1",
                "|switch|p1a: Venusaur|Venusaur, L80|100/100",
                "|switch|p2a: Starmie|Starmie, L80|100/100 par",
                "|turn|1",
            ],
        )
        metadata = {
            "Thunder Wave": _metadata(
                "Thunder Wave",
                category="Status",
                base_power=0,
                accuracy=100,
                status="par",
            )
        }

        with _patched_calc(metadata, {}):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(4))

        self.assertEqual(plan.actions, [])
        self.assertEqual(plan.fallback_reason, "all heuristic scores were zero")

    def test_hyper_beam_is_penalized_when_it_does_not_ko(self) -> None:
        context = _context(
            [
                {"move": "Hyper Beam", "id": "hyperbeam", "disabled": False},
                {"move": "Body Slam", "id": "bodyslam", "disabled": False},
            ]
        )
        metadata = {
            "Hyper Beam": _metadata("Hyper Beam", base_power=150, accuracy=90, recharge=True),
            "Body Slam": _metadata("Body Slam", base_power=85),
        }
        ranges = {"Hyper Beam": (60, 70), "Body Slam": (35, 45)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(4))

        by_choice = {action.choice: action for action in plan.actions}
        self.assertLess(by_choice["move 1"].score, by_choice["move 2"].score)

    def test_hyper_beam_is_partially_penalized_when_ko_is_uncertain(self) -> None:
        context = _context(
            [
                {"move": "Hyper Beam", "id": "hyperbeam", "disabled": False},
                {"move": "Body Slam", "id": "bodyslam", "disabled": False},
            ]
        )
        metadata = {
            "Hyper Beam": _metadata("Hyper Beam", base_power=150, accuracy=100, recharge=True),
            "Body Slam": _metadata("Body Slam", base_power=85),
        }
        ranges = {"Hyper Beam": (70, 110), "Body Slam": (65, 65)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(5))

        by_choice = {action.choice: action for action in plan.actions}
        self.assertLess(by_choice["move 1"].score, by_choice["move 2"].score)

    def test_high_crit_moves_receive_reliability_bonus(self) -> None:
        context = _context(
            [
                {"move": "Slash", "id": "slash", "disabled": False},
                {"move": "Strength", "id": "strength", "disabled": False},
            ]
        )
        metadata = {
            "Slash": _metadata("Slash", base_power=70, high_crit=True),
            "Strength": _metadata("Strength", base_power=80),
        }
        ranges = {"Slash": (40, 40), "Strength": (40, 40)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(6))

        by_choice = {action.choice: action for action in plan.actions}
        self.assertGreater(by_choice["move 1"].score, by_choice["move 2"].score)

    def test_charge_moves_are_penalized(self) -> None:
        context = _context(
            [
                {"move": "Solar Beam", "id": "solarbeam", "disabled": False},
                {"move": "Razor Leaf", "id": "razorleaf", "disabled": False},
            ]
        )
        metadata = {
            "Solar Beam": _metadata("Solar Beam", base_power=120, charge=True),
            "Razor Leaf": _metadata("Razor Leaf", base_power=55, high_crit=True),
        }
        ranges = {"Solar Beam": (70, 80), "Razor Leaf": (35, 45)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(5))

        by_choice = {action.choice: action for action in plan.actions}
        self.assertLess(by_choice["move 1"].score, 40)
        self.assertGreater(by_choice["move 2"].score, by_choice["move 1"].score)

    def test_amnesia_is_high_value_at_high_hp_and_low_value_at_low_hp(self) -> None:
        high_hp = _context(
            [{"move": "Amnesia", "id": "amnesia", "disabled": False}],
            active_condition="90/100",
        )
        low_hp = _context(
            [{"move": "Amnesia", "id": "amnesia", "disabled": False}],
            active_condition="20/100",
        )
        metadata = {
            "Amnesia": _metadata(
                "Amnesia",
                category="Status",
                base_power=0,
                accuracy=True,
                boosts={"spa": 2, "spd": 2},
            )
        }

        with _patched_calc(metadata, {}):
            high_plan = build_custom_bot_plan(high_hp, project_root=Path.cwd(), rng=random.Random(6))
            low_plan = build_custom_bot_plan(low_hp, project_root=Path.cwd(), rng=random.Random(6))

        self.assertGreater(high_plan.actions[0].score, 70)
        self.assertLess(low_plan.actions[0].score, 5)

    def test_freeze_and_evasion_are_high_value_in_gen1(self) -> None:
        context = _context(
            [{"move": "Double Team", "id": "doubleteam", "disabled": False}],
        )
        metadata = {
            "Double Team": _metadata(
                "Double Team",
                category="Status",
                base_power=0,
                accuracy=True,
                boosts={"evasion": 1},
            )
        }

        with _patched_calc(metadata, {}):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(7))

        self.assertEqual(_status_base_value("frz", sleep_clause_active=False), 88.0)
        self.assertGreater(plan.actions[0].score, 70)

    def test_switches_are_added_when_active_moves_are_weak(self) -> None:
        context = _context(
            [{"move": "Tackle", "id": "tackle", "disabled": False}],
            bench=[
                {"ident": "p1: Jolteon", "details": "Jolteon, L80", "condition": "100/100"},
            ],
        )
        metadata = {"Tackle": _metadata("Tackle", base_power=40)}
        ranges = {"Tackle": (5, 7)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(7))

        self.assertIn("switch 2", {action.choice for action in plan.actions})

    def test_low_hp_active_can_still_be_sacrificed_without_clear_switch(self) -> None:
        context = _context(
            [{"move": "Tackle", "id": "tackle", "disabled": False}],
            active_condition="10/100",
            bench=[
                {"ident": "p1: Jolteon", "details": "Jolteon, L80", "condition": "50/100"},
            ],
        )
        metadata = {"Tackle": _metadata("Tackle", base_power=40)}
        ranges = {"Tackle": (5, 7)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(8))

        self.assertNotIn("switch 2", {action.choice for action in plan.actions})

    def test_unsupported_format_falls_back_to_legal_choice(self) -> None:
        context = _context(
            [{"move": "Surf", "id": "surf", "disabled": False}],
            format_name="gen3randombattle",
        )

        plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(9))

        self.assertEqual(plan.decision, "move 1")
        self.assertEqual(plan.fallback_reason, "unsupported format or request kind")

    def test_forced_switch_plan_scores_available_switches(self) -> None:
        context = _context([])
        context["request_kind"] = "switch"
        context["request"] = {
            "forceSwitch": [True],
            "side": {
                "pokemon": [
                    {"ident": "p1: Venusaur", "details": "Venusaur, L80", "condition": "0 fnt", "active": True},
                    {"ident": "p1: Jolteon", "details": "Jolteon, L80", "condition": "90/100"},
                    {"ident": "p1: Snorlax", "details": "Snorlax, L80", "condition": "40/100"},
                ]
            },
        }

        plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(10))

        self.assertIn(plan.decision, {"switch 2", "switch 3"})
        self.assertGreater(
            {action.choice: action for action in plan.actions}["switch 2"].score,
            {action.choice: action for action in plan.actions}["switch 3"].score,
        )

    def test_choose_weighted_tracks_weight_distribution(self) -> None:
        actions = [
            _scored_action("move 1", "Weak", 1.0, "weak"),
            _scored_action("move 2", "Strong", 3.0, "strong"),
        ]
        rng = random.Random(123)
        counts = {"move 1": 0, "move 2": 0}

        for _ in range(1000):
            counts[_choose_weighted(actions, rng)] += 1

        self.assertGreater(counts["move 2"], 850)
        self.assertLess(counts["move 2"], 950)

    def test_no_enabled_moves_falls_back(self) -> None:
        context = _context(
            [{"move": "Surf", "id": "surf", "disabled": True}],
        )

        plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(11))

        self.assertEqual(plan.fallback_reason, "no enabled moves")

    def test_sleep_clause_detection_from_public_log_lines(self) -> None:
        context = _context(
            [{"move": "Sleep Powder", "id": "sleeppowder", "disabled": False}],
        )
        public_lines = [
            "|gen|1",
            "|switch|p1a: Venusaur|Venusaur, L80|100/100",
            "|switch|p2a: Starmie|Starmie, L80|100/100",
            "|move|p1a: Venusaur|Sleep Powder|p2a: Starmie",
            "|-status|p2a: Starmie|slp",
        ]

        self.assertTrue(_sleep_clause_active(context, public_lines))

    def test_revealed_opponent_moves_extracts_unique_moves(self) -> None:
        context = _context(
            [{"move": "Tackle", "id": "tackle", "disabled": False}],
        )
        public_lines = [
            "|move|p2a: Starmie|Surf|p1a: Venusaur",
            "|move|p1a: Venusaur|Sleep Powder|p2a: Starmie",
            "|move|p2a: Starmie|Surf|p1a: Venusaur",
            "|move|p2a: Starmie|Thunder Wave|p1a: Venusaur",
        ]

        self.assertEqual(_revealed_opponent_moves(context, public_lines), ["Surf", "Thunder Wave"])

    def test_damage_calc_failure_is_visible_in_notes(self) -> None:
        context = _context(
            [{"move": "Tackle", "id": "tackle", "disabled": False}],
        )
        metadata = {"Tackle": _metadata("Tackle", base_power=40)}

        with _patched_calc(metadata, {}, damage_error=ConfigError("worker down")):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(12))

        self.assertIn("damage calc batch failed", plan.notes)
        self.assertIn("damage calc batch failed", plan.warnings[0])


def _context(
    moves: list[dict],
    *,
    format_name: str = "gen1randombattle",
    active_condition: str = "100/100",
    bench: list[dict] | None = None,
    recent_public_events: list[str] | None = None,
) -> dict:
    side_pokemon = [
        {
            "ident": "p1: Venusaur",
            "details": "Venusaur, L80",
            "condition": active_condition,
            "active": True,
            "stats": {"hp": 280, "atk": 180, "def": 180, "spa": 200, "spd": 200, "spe": 180},
        }
    ]
    side_pokemon.extend(bench or [])
    request = {
        "active": [{"moves": moves}],
        "side": {"pokemon": side_pokemon},
    }
    return {
        "schema_version": "pokerena.turn-context.v1",
        "battle_id": "battle-gen1",
        "agent_id": "custom",
        "provider": "pokerena-custom-bot",
        "player_slot": "p1",
        "context_token": "battle-gen1:p1:1:1",
        "format_name": format_name,
        "turn_number": 1,
        "phase": "turn",
        "request_kind": "move",
        "rqid": "1",
        "request_sequence": 1,
        "decision_attempt": 1,
        "signals": {"decision_required": True},
        "legal_action_hints": [f"move {index}" for index in range(1, len(moves) + 1)],
        "request": request,
        "side": request["side"],
        "active": request["active"],
        "recent_public_events": recent_public_events
        or [
            "|gen|1",
            "|switch|p1a: Venusaur|Venusaur, L80|100/100",
            "|switch|p2a: Starmie|Starmie, L80|100/100",
            "|turn|1",
        ],
        "last_error": None,
    }


def _metadata(
    name: str,
    *,
    category: str = "Physical",
    base_power: int = 50,
    accuracy: int | bool = 100,
    status: str | None = None,
    boosts: dict | None = None,
    secondary: dict | None = None,
    recharge: bool = False,
    charge: bool = False,
    high_crit: bool = False,
) -> dict:
    return {
        "requested_name": name,
        "id": name.lower().replace(" ", ""),
        "name": name,
        "category": category,
        "base_power": base_power,
        "accuracy": accuracy,
        "status": status,
        "volatile_status": None,
        "boosts": boosts or {},
        "self": {},
        "flags": {},
        "secondary": secondary or {},
        "crit_ratio": 2 if high_crit else 1,
        "high_crit": high_crit,
        "recharge": recharge,
        "charge": charge,
        "target": "normal",
    }


def _patched_calc(
    metadata: dict[str, dict],
    ranges: dict[str, tuple[int, int]],
    *,
    damage_error: Exception | None = None,
):
    def fake_metadata(*, move_names, **kwargs):
        return {
            "schema_version": "pokerena.move-metadata-result.v1",
            "generation": 1,
            "moves": [metadata[name] for name in move_names],
        }

    def fake_damage_batch(payload, **kwargs):
        if damage_error is not None:
            raise damage_error
        results = []
        for request in payload["requests"]:
            move_name = request["move"]["name"]
            min_pct, max_pct = ranges.get(move_name, (0, 0))
            results.append(
                {
                    "status": "ok" if max_pct > 0 else "skipped",
                    "skip_reason": "non_damaging",
                    "move_name": move_name,
                    "generation": 1,
                    "result": {
                        "range_percent": {"min": min_pct, "max": max_pct},
                        "knockout": {"chance": None, "hits": None, "text": ""},
                    },
                }
            )
        return {"schema_version": "pokerena.damage-batch-result.v1", "results": results}

    return mock.patch.multiple(
        "pokerena.custom_bot",
        describe_move_metadata=mock.Mock(side_effect=fake_metadata),
        run_damage_calc_batch=mock.Mock(side_effect=fake_damage_batch),
    )
