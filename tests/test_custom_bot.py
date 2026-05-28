from __future__ import annotations

import random
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pokerena.config import ConfigError
from pokerena.custom_bot import (
    PokemonState,
    _base_speed_for_species,
    _choose_weighted,
    _effective_player_slot,
    _pokemon_types,
    _revealed_opponent_moves,
    _scored_action,
    _selection_pool,
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

    def test_selection_pool_prunes_dominated_low_score_moves(self) -> None:
        actions = [
            _scored_action("move 1", "Earthquake", 28.8, "damage"),
            _scored_action("move 2", "Slash", 26.9, "damage"),
            _scored_action("move 4", "Rock Slide", 12.9, "damage"),
        ]

        pool = _selection_pool(actions)

        self.assertEqual([action.choice for action in pool], ["move 1", "move 2"])

    def test_dominated_damage_move_is_not_sampled(self) -> None:
        context = _context(
            [
                {"move": "Earthquake", "id": "earthquake", "disabled": False},
                {"move": "Slash", "id": "slash", "disabled": False},
                {"move": "Substitute", "id": "substitute", "disabled": False},
                {"move": "Rock Slide", "id": "rockslide", "disabled": False},
            ],
            own_species="Dugtrio",
            opponent_species="Dugtrio",
        )
        metadata = {
            "Earthquake": _metadata("Earthquake", base_power=100),
            "Slash": _metadata("Slash", base_power=70, high_crit=True),
            "Substitute": _metadata("Substitute", category="Status", base_power=0),
            "Rock Slide": _metadata("Rock Slide", base_power=75, accuracy=90),
        }
        ranges = {"Earthquake": (26.4, 31.1), "Slash": (23.4, 27.8), "Rock Slide": (13.0, 15.7)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(2))

        self.assertEqual({action.choice for action in plan.actions}, {"move 1", "move 2"})
        self.assertNotEqual(plan.decision, "move 4")

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
        self.assertIn("move 2", by_choice)
        self.assertNotIn("move 1", by_choice)

    def test_status_move_is_deprioritized_when_ko_is_available(self) -> None:
        context = _context(
            [
                {"move": "Body Slam", "id": "bodyslam", "disabled": False},
                {"move": "Sleep Powder", "id": "sleeppowder", "disabled": False},
            ],
            recent_public_events=[
                "|gen|1",
                "|switch|p1a: Venusaur|Venusaur, L80|100/100",
                "|switch|p2a: Starmie|Starmie, L80|30/100",
                "|turn|3",
            ],
        )
        metadata = {
            "Body Slam": _metadata("Body Slam", base_power=85),
            "Sleep Powder": _metadata(
                "Sleep Powder",
                category="Status",
                base_power=0,
                accuracy=75,
                status="slp",
            ),
        }
        ranges = {"Body Slam": (35, 45)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(2))

        self.assertEqual({action.choice for action in plan.actions}, {"move 1"})

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

    def test_thunder_wave_scores_zero_against_ground_type(self) -> None:
        context = _context(
            [
                {"move": "Tackle", "id": "tackle", "disabled": False},
                {"move": "Thunder Wave", "id": "thunderwave", "disabled": False},
            ],
            opponent_species="Dugtrio",
        )
        metadata = {
            "Tackle": _metadata("Tackle", base_power=40),
            "Thunder Wave": _metadata(
                "Thunder Wave",
                category="Status",
                base_power=0,
                accuracy=100,
                status="par",
            ),
        }
        ranges = {"Tackle": (12, 15)}

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_pokedex_fixture(project_root)
            with _patched_calc(metadata, ranges):
                plan = build_custom_bot_plan(context, project_root=project_root, rng=random.Random(4))

        self.assertEqual({action.choice for action in plan.actions}, {"move 1"})
        self.assertNotEqual(plan.decision, "move 2")

    def test_zero_effective_damage_move_is_not_selected_when_calc_skips(self) -> None:
        context = _context(
            [
                {"move": "Thunderbolt", "id": "thunderbolt", "disabled": False},
                {"move": "Body Slam", "id": "bodyslam", "disabled": False},
            ],
            opponent_species="Marowak",
        )
        metadata = {
            "Thunderbolt": _metadata("Thunderbolt", base_power=95, move_type="Electric"),
            "Body Slam": _metadata("Body Slam", base_power=85),
        }
        ranges = {"Body Slam": (20, 25)}

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_pokedex_fixture(project_root)
            with _patched_calc(metadata, ranges):
                plan = build_custom_bot_plan(context, project_root=project_root, rng=random.Random(4))

        self.assertEqual({action.choice for action in plan.actions}, {"move 2"})
        self.assertNotEqual(plan.decision, "move 1")

    def test_zero_effective_damage_move_is_not_selected_when_calc_batch_fails(self) -> None:
        context = _context(
            [
                {"move": "Thunderbolt", "id": "thunderbolt", "disabled": False},
                {"move": "Body Slam", "id": "bodyslam", "disabled": False},
            ],
            opponent_species="Marowak",
        )
        metadata = {
            "Thunderbolt": _metadata("Thunderbolt", base_power=95, move_type="Electric"),
            "Body Slam": _metadata("Body Slam", base_power=85),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_pokedex_fixture(project_root)
            with _patched_calc(metadata, {}, damage_error=ConfigError("worker down")):
                plan = build_custom_bot_plan(context, project_root=project_root, rng=random.Random(4))

        self.assertEqual({action.choice for action in plan.actions}, {"move 2"})
        self.assertIn("damage calc batch failed", plan.warnings[0])
        self.assertNotEqual(plan.decision, "move 1")

    def test_pokedex_type_lookup_parses_showdown_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_pokedex_fixture(project_root)

            dugtrio = PokemonState("Dugtrio", None, {"species": "Dugtrio"}, 1.0, None)
            marowak = PokemonState("Marowak", None, {"species": "Marowak"}, 1.0, None)

            self.assertEqual(_pokemon_types(dugtrio, project_root), ["Ground"])
            self.assertEqual(_pokemon_types(marowak, project_root), ["Ground"])
            self.assertEqual(_base_speed_for_species("Dugtrio", project_root), 120)

    def test_confuse_ray_scores_zero_when_target_already_confused(self) -> None:
        context = _context(
            [
                {"move": "Tackle", "id": "tackle", "disabled": False},
                {"move": "Confuse Ray", "id": "confuseray", "disabled": False},
            ],
            opponent_species="Hypno",
            recent_public_events=[
                "|gen|1",
                "|switch|p1a: Golbat|Golbat, L80|100/100",
                "|switch|p2a: Hypno|Hypno, L80|100/100",
                "|-start|p2a: Hypno|confusion",
                "|turn|2",
            ],
        )
        metadata = {
            "Tackle": _metadata("Tackle", base_power=40),
            "Confuse Ray": _metadata(
                "Confuse Ray",
                category="Status",
                base_power=0,
                accuracy=100,
                volatile_status="confusion",
            ),
        }
        ranges = {"Tackle": (12, 15)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(4))

        self.assertEqual({action.choice for action in plan.actions}, {"move 1"})
        self.assertNotEqual(plan.decision, "move 2")

    def test_confuse_ray_is_deprioritized_when_ko_is_available(self) -> None:
        context = _context(
            [
                {"move": "Body Slam", "id": "bodyslam", "disabled": False},
                {"move": "Confuse Ray", "id": "confuseray", "disabled": False},
            ],
            opponent_species="Hypno",
            recent_public_events=[
                "|gen|1",
                "|switch|p1a: Golbat|Golbat, L80|100/100",
                "|switch|p2a: Hypno|Hypno, L80|30/100",
                "|turn|2",
            ],
        )
        metadata = {
            "Body Slam": _metadata("Body Slam", base_power=85),
            "Confuse Ray": _metadata(
                "Confuse Ray",
                category="Status",
                base_power=0,
                accuracy=100,
                volatile_status="confusion",
            ),
        }
        ranges = {"Body Slam": (35, 45)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(4))

        self.assertEqual({action.choice for action in plan.actions}, {"move 1"})

    def test_confuse_ray_scores_after_confusion_ends(self) -> None:
        context = _context(
            [{"move": "Confuse Ray", "id": "confuseray", "disabled": False}],
            opponent_species="Hypno",
            recent_public_events=[
                "|gen|1",
                "|switch|p1a: Golbat|Golbat, L80|100/100",
                "|switch|p2a: Hypno|Hypno, L80|100/100",
                "|-start|p2a: Hypno|confusion",
                "|-end|p2a: Hypno|confusion",
                "|turn|4",
            ],
        )
        metadata = {
            "Confuse Ray": _metadata(
                "Confuse Ray",
                category="Status",
                base_power=0,
                accuracy=100,
                volatile_status="confusion",
            )
        }

        with _patched_calc(metadata, {}):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(4))

        self.assertEqual({action.choice for action in plan.actions}, {"move 1"})
        self.assertEqual(plan.actions[0].reason, "confusion pressure")

    def test_status_scoring_uses_request_side_when_context_slot_is_stale(self) -> None:
        context = _context(
            [
                {"move": "Tackle", "id": "tackle", "disabled": False},
                {"move": "Stun Spore", "id": "stunspore", "disabled": False},
            ],
            player_slot="p1",
            side_id="p2",
            own_species="Butterfree",
            opponent_species="Exeggcute",
            recent_public_events=[
                "|gen|1",
                "|switch|p1a: Exeggcute|Exeggcute, L84|100/100",
                "|switch|p2a: Butterfree|Butterfree, L77|100/100",
                "|-status|p1a: Exeggcute|par",
                "|turn|10",
            ],
        )
        metadata = {
            "Tackle": _metadata("Tackle", base_power=40),
            "Stun Spore": _metadata(
                "Stun Spore",
                category="Status",
                base_power=0,
                accuracy=75,
                status="par",
            ),
        }
        ranges = {"Tackle": (12, 16)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(13))

        self.assertEqual(_effective_player_slot(context), "p2")
        self.assertIn("move 1", {action.choice for action in plan.actions})
        self.assertNotIn("move 2", {action.choice for action in plan.actions})

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
        self.assertIn("move 2", by_choice)
        self.assertNotIn("move 1", by_choice)

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

    def test_explosion_is_penalized_from_high_hp_without_ko(self) -> None:
        context = _context(
            [
                {"move": "Explosion", "id": "explosion", "disabled": False},
                {"move": "Thunderbolt", "id": "thunderbolt", "disabled": False},
            ],
            own_species="Gengar",
            opponent_species="Starmie",
        )
        metadata = {
            "Explosion": _metadata("Explosion", base_power=170),
            "Thunderbolt": _metadata("Thunderbolt", base_power=95, move_type="Electric"),
        }
        ranges = {"Explosion": (65, 75), "Thunderbolt": (35, 45)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(5))

        self.assertEqual({action.choice for action in plan.actions}, {"move 2"})

    def test_explosion_is_penalized_from_high_hp_when_ko_trade_is_not_ahead(self) -> None:
        context = _context(
            [
                {"move": "Explosion", "id": "explosion", "disabled": False},
                {"move": "Thunderbolt", "id": "thunderbolt", "disabled": False},
            ],
            own_species="Gengar",
            opponent_species="Starmie",
            recent_public_events=[
                "|gen|1",
                "|switch|p1a: Gengar|Gengar, L80|100/100",
                "|switch|p2a: Starmie|Starmie, L80|60/100",
                "|turn|4",
            ],
        )
        metadata = {
            "Explosion": _metadata("Explosion", base_power=170),
            "Thunderbolt": _metadata("Thunderbolt", base_power=95, move_type="Electric"),
        }
        ranges = {"Explosion": (70, 80), "Thunderbolt": (35, 45)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(5))

        self.assertEqual({action.choice for action in plan.actions}, {"move 2"})

    def test_explosion_is_allowed_for_ko_when_ahead_in_mon_count(self) -> None:
        context = _context(
            [
                {"move": "Explosion", "id": "explosion", "disabled": False},
                {"move": "Thunderbolt", "id": "thunderbolt", "disabled": False},
            ],
            own_species="Gengar",
            opponent_species="Starmie",
            bench=[
                {"ident": "p1: Tauros", "details": "Tauros, L80", "condition": "100/100"},
            ],
            recent_public_events=[
                "|gen|1",
                "|switch|p1a: Gengar|Gengar, L80|100/100",
                "|switch|p2a: Alakazam|Alakazam, L80|0 fnt",
                "|faint|p2a: Alakazam",
                "|switch|p2a: Starmie|Starmie, L80|60/100",
                "|turn|4",
            ],
        )
        metadata = {
            "Explosion": _metadata("Explosion", base_power=170),
            "Thunderbolt": _metadata("Thunderbolt", base_power=95, move_type="Electric"),
        }
        ranges = {"Explosion": (70, 80), "Thunderbolt": (35, 45)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(5))

        self.assertEqual({action.choice for action in plan.actions}, {"move 1"})

    def test_explosion_is_allowed_from_low_hp(self) -> None:
        context = _context(
            [
                {"move": "Explosion", "id": "explosion", "disabled": False},
                {"move": "Thunderbolt", "id": "thunderbolt", "disabled": False},
            ],
            own_species="Gengar",
            opponent_species="Starmie",
            active_condition="20/100",
        )
        metadata = {
            "Explosion": _metadata("Explosion", base_power=170),
            "Thunderbolt": _metadata("Thunderbolt", base_power=95, move_type="Electric"),
        }
        ranges = {"Explosion": (65, 75), "Thunderbolt": (20, 25)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(5))

        self.assertEqual({action.choice for action in plan.actions}, {"move 1"})

    def test_explosion_is_penalized_when_calc_batch_fails(self) -> None:
        context = _context(
            [
                {"move": "Explosion", "id": "explosion", "disabled": False},
                {"move": "Body Slam", "id": "bodyslam", "disabled": False},
            ],
            own_species="Gengar",
            opponent_species="Starmie",
        )
        metadata = {
            "Explosion": _metadata("Explosion", base_power=170),
            "Body Slam": _metadata("Body Slam", base_power=85),
        }

        with _patched_calc(metadata, {}, damage_error=ConfigError("worker down")):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(5))

        self.assertEqual({action.choice for action in plan.actions}, {"move 2"})

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

    def test_repeat_setup_boost_is_deprioritized_when_ko_is_available(self) -> None:
        context = _context(
            [
                {"move": "Body Slam", "id": "bodyslam", "disabled": False},
                {"move": "Swords Dance", "id": "swordsdance", "disabled": False},
            ],
            recent_public_events=[
                "|gen|1",
                "|switch|p1a: Venusaur|Venusaur, L80|100/100",
                "|-boost|p1a: Venusaur|atk|2",
                "|switch|p2a: Starmie|Starmie, L80|30/100",
                "|turn|4",
            ],
        )
        metadata = {
            "Body Slam": _metadata("Body Slam", base_power=85),
            "Swords Dance": _metadata(
                "Swords Dance",
                category="Status",
                base_power=0,
                accuracy=True,
                boosts={"atk": 2},
            ),
        }
        ranges = {"Body Slam": (35, 45)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(7))

        self.assertEqual({action.choice for action in plan.actions}, {"move 1"})

    def test_agility_scores_zero_when_already_faster(self) -> None:
        context = _context(
            [{"move": "Agility", "id": "agility", "disabled": False}],
            own_stats={"hp": 280, "atk": 180, "def": 180, "spa": 200, "spd": 200, "spe": 220},
        )
        metadata = {
            "Agility": _metadata(
                "Agility",
                category="Status",
                base_power=0,
                accuracy=True,
                boosts={"spe": 2},
            )
        }

        with _patched_calc(metadata, {}), mock.patch(
            "pokerena.custom_bot._estimated_gen1_speed",
            return_value=160,
        ):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(7))

        self.assertEqual(plan.actions, [])
        self.assertEqual(plan.fallback_reason, "all heuristic scores were zero")

    def test_agility_scores_high_when_slower_at_high_hp(self) -> None:
        context = _context(
            [{"move": "Agility", "id": "agility", "disabled": False}],
            own_stats={"hp": 280, "atk": 180, "def": 180, "spa": 200, "spd": 200, "spe": 120},
        )
        metadata = {
            "Agility": _metadata(
                "Agility",
                category="Status",
                base_power=0,
                accuracy=True,
                boosts={"spe": 2},
            )
        }

        with _patched_calc(metadata, {}), mock.patch(
            "pokerena.custom_bot._estimated_gen1_speed",
            return_value=180,
        ):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(7))

        self.assertGreater(plan.actions[0].score, 70)
        self.assertIn("outspeed likely faster opponent", plan.actions[0].reason)

    def test_agility_offsets_paralysis_until_speed_boost_lands(self) -> None:
        paralyzed_context = _context(
            [{"move": "Agility", "id": "agility", "disabled": False}],
            own_species="Jolteon",
            active_condition="80/100 par",
            own_stats={"hp": 280, "atk": 180, "def": 180, "spa": 200, "spd": 200, "spe": 240},
        )
        boosted_context = _context(
            [{"move": "Agility", "id": "agility", "disabled": False}],
            own_species="Jolteon",
            active_condition="80/100 par",
            own_stats={"hp": 280, "atk": 180, "def": 180, "spa": 200, "spd": 200, "spe": 240},
            recent_public_events=[
                "|gen|1",
                "|switch|p1a: Jolteon|Jolteon, L80|80/100 par",
                "|switch|p2a: Snorlax|Snorlax, L80|100/100",
                "|-boost|p1a: Jolteon|spe|2",
                "|turn|4",
            ],
        )
        metadata = {
            "Agility": _metadata(
                "Agility",
                category="Status",
                base_power=0,
                accuracy=True,
                boosts={"spe": 2},
            )
        }

        with _patched_calc(metadata, {}), mock.patch(
            "pokerena.custom_bot._estimated_gen1_speed",
            return_value=120,
        ):
            paralyzed_plan = build_custom_bot_plan(
                paralyzed_context,
                project_root=Path.cwd(),
                rng=random.Random(7),
            )
            boosted_plan = build_custom_bot_plan(
                boosted_context,
                project_root=Path.cwd(),
                rng=random.Random(7),
            )

        self.assertGreater(paralyzed_plan.actions[0].score, 70)
        self.assertIn("paralysis Speed loss", paralyzed_plan.actions[0].reason)
        self.assertEqual(boosted_plan.actions, [])
        self.assertEqual(boosted_plan.fallback_reason, "all heuristic scores were zero")

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
            recent_public_events=[
                "|gen|1",
                "|switch|p1a: Venusaur|Venusaur, L80|100/100",
                "|switch|p2a: Starmie|Starmie, L80|100/100",
                "|move|p1a: Venusaur|Tackle|p2a: Starmie",
                "|move|p1a: Venusaur|Tackle|p2a: Starmie",
                "|turn|3",
            ],
        )
        metadata = {"Tackle": _metadata("Tackle", base_power=40)}
        ranges = {"Tackle": (5, 7)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(7))

        self.assertIn("switch 2", {action.choice for action in plan.actions})

    def test_switches_are_suppressed_without_known_threat_when_active_move_is_usable(self) -> None:
        context = _context(
            [{"move": "Surf", "id": "surf", "disabled": False}],
            bench=[
                {"ident": "p1: Jolteon", "details": "Jolteon, L80", "condition": "100/100"},
                {"ident": "p1: Snorlax", "details": "Snorlax, L80", "condition": "100/100"},
            ],
        )
        metadata = {"Surf": _metadata("Surf", base_power=95)}
        ranges = {"Surf": (30, 35)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(7))

        self.assertNotIn("switch 2", {action.choice for action in plan.actions})
        self.assertNotIn("switch 3", {action.choice for action in plan.actions})

    def test_switches_are_suppressed_right_after_forced_switch_with_usable_move(self) -> None:
        context = _context(
            [{"move": "Blizzard", "id": "blizzard", "disabled": False}],
            player_slot="p2",
            side_id="p2",
            own_species="Articuno",
            opponent_species="Rhydon",
            bench=[
                {
                    "ident": "p2: Vaporeon",
                    "details": "Vaporeon, L74",
                    "condition": "100/100",
                    "moves": ["surf", "blizzard", "rest", "bodyslam"],
                },
            ],
            recent_public_events=[
                "|gen|1",
                "|switch|p1a: Rhydon|Rhydon, L80|100/100",
                "|switch|p2a: Drowzee|Drowzee, L84|0 fnt",
                "|faint|p2a: Drowzee",
                "|switch|p2a: Articuno|Articuno, L70|100/100",
                "|turn|10",
            ],
        )
        metadata = {"Blizzard": _metadata("Blizzard", base_power=120)}
        ranges = {"Blizzard": (30, 35), "surf": (90, 100)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(7))

        self.assertIn("move 1", {action.choice for action in plan.actions})
        self.assertNotIn("switch 2", {action.choice for action in plan.actions})

    def test_switches_are_suppressed_immediately_after_any_switch_in(self) -> None:
        context = _context(
            [{"move": "Slash", "id": "slash", "disabled": False}],
            player_slot="p2",
            side_id="p2",
            own_species="Meowth",
            opponent_species="Golem",
            bench=[
                {
                    "ident": "p2: Shellder",
                    "details": "Shellder, L90",
                    "condition": "100/100",
                    "moves": ["surf", "blizzard", "explosion", "doubleedge"],
                },
            ],
            recent_public_events=[
                "|gen|1",
                "|switch|p1a: Golem|Golem, L71|100/100",
                "|switch|p2a: Jolteon|Jolteon, L69|0 fnt",
                "|faint|p2a: Jolteon",
                "|switch|p2a: Meowth|Meowth, L85|100/100",
                "|turn|2",
            ],
        )
        metadata = {"Slash": _metadata("Slash", base_power=70, high_crit=True)}
        ranges = {"Slash": (0, 0), "surf": (140, 160)}

        with _patched_calc(metadata, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(7))

        self.assertIn("move 1", {action.choice for action in plan.actions})
        self.assertNotIn("switch 2", {action.choice for action in plan.actions})

    def test_active_boosts_downregulate_voluntary_switching(self) -> None:
        common_events = [
            "|gen|1",
            "|switch|p1a: Venusaur|Venusaur, L80|100/100",
            "|switch|p2a: Starmie|Starmie, L80|100/100",
            "|move|p1a: Venusaur|Tackle|p2a: Starmie",
            "|move|p2a: Starmie|Surf|p1a: Venusaur",
            "|move|p1a: Venusaur|Tackle|p2a: Starmie",
            "|move|p2a: Starmie|Surf|p1a: Venusaur",
            "|turn|3",
        ]
        unboosted = _context(
            [{"move": "Tackle", "id": "tackle", "disabled": False}],
            bench=[
                {"ident": "p1: Jolteon", "details": "Jolteon, L80", "condition": "100/100"},
            ],
            recent_public_events=common_events,
        )
        boosted = _context(
            [{"move": "Tackle", "id": "tackle", "disabled": False}],
            bench=[
                {"ident": "p1: Jolteon", "details": "Jolteon, L80", "condition": "100/100"},
            ],
            recent_public_events=[
                common_events[0],
                common_events[1],
                common_events[2],
                "|-boost|p1a: Venusaur|atk|2",
                *common_events[3:],
            ],
        )
        metadata = {"Tackle": _metadata("Tackle", base_power=40)}
        ranges = {"Tackle": (12, 15), "Surf": (50, 50)}

        with _patched_calc(metadata, ranges):
            unboosted_plan = build_custom_bot_plan(unboosted, project_root=Path.cwd(), rng=random.Random(7))
            boosted_plan = build_custom_bot_plan(boosted, project_root=Path.cwd(), rng=random.Random(7))

        self.assertIn("switch 2", {action.choice for action in unboosted_plan.actions})
        self.assertNotIn("switch 2", {action.choice for action in boosted_plan.actions})

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

    def test_forced_switch_plan_scores_available_switches_by_matchup(self) -> None:
        context = _context([])
        context["request_kind"] = "switch"
        context["request"] = {
            "forceSwitch": [True],
            "side": {
                "pokemon": [
                    {"ident": "p1: Venusaur", "details": "Venusaur, L80", "condition": "0 fnt", "active": True},
                    {
                        "ident": "p1: Vaporeon",
                        "details": "Vaporeon, L74",
                        "condition": "100/100",
                        "moves": ["surf", "blizzard", "rest", "bodyslam"],
                    },
                    {
                        "ident": "p1: Lickitung",
                        "details": "Lickitung, L80",
                        "condition": "100/100",
                        "moves": ["bodyslam", "stomp", "screech", "rest"],
                    },
                ]
            },
        }
        context["side"] = context["request"]["side"]
        context["recent_public_events"] = [
            "|gen|1",
            "|switch|p1a: Venusaur|Venusaur, L80|0 fnt",
            "|faint|p1a: Venusaur",
            "|switch|p2a: Rhydon|Rhydon, L80|100/100",
            "|turn|8",
        ]
        ranges = {"surf": (95, 110), "blizzard": (30, 35), "bodyslam": (12, 16), "stomp": (10, 13)}

        with _patched_calc({}, ranges):
            plan = build_custom_bot_plan(context, project_root=Path.cwd(), rng=random.Random(10))

        self.assertEqual(plan.decision, "switch 2")
        self.assertEqual({action.choice for action in plan.actions}, {"switch 2"})

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
    player_slot: str = "p1",
    side_id: str = "p1",
    own_species: str = "Venusaur",
    opponent_species: str = "Starmie",
    active_condition: str = "100/100",
    own_stats: dict | None = None,
    bench: list[dict] | None = None,
    recent_public_events: list[str] | None = None,
) -> dict:
    side_pokemon = [
        {
            "ident": f"{side_id}: {own_species}",
            "details": f"{own_species}, L80",
            "condition": active_condition,
            "active": True,
            "stats": own_stats
            or {"hp": 280, "atk": 180, "def": 180, "spa": 200, "spd": 200, "spe": 180},
        }
    ]
    side_pokemon.extend(bench or [])
    request = {
        "active": [{"moves": moves}],
        "side": {"id": side_id, "pokemon": side_pokemon},
    }
    opponent_slot = "p2" if side_id == "p1" else "p1"
    return {
        "schema_version": "pokerena.turn-context.v1",
        "battle_id": "battle-gen1",
        "agent_id": "custom",
        "provider": "pokerena-custom-bot",
        "player_slot": player_slot,
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
            f"|switch|{side_id}a: {own_species}|{own_species}, L80|100/100",
            f"|switch|{opponent_slot}a: {opponent_species}|{opponent_species}, L80|100/100",
            "|turn|1",
        ],
        "last_error": None,
    }


def _write_pokedex_fixture(project_root: Path) -> None:
    pokedex_path = project_root / "vendor" / "pokemon-showdown" / "dist" / "data" / "pokedex.js"
    pokedex_path.parent.mkdir(parents=True)
    pokedex_path.write_text(
        """
exports.Pokedex = {
  dugtrio: {
    num: 51,
    name: "Dugtrio",
    types: ["Ground"],
    baseStats: { hp: 35, atk: 100, def: 50, spa: 50, spd: 70, spe: 120 },
    abilities: { 0: "Sand Veil" },
  },
  starmie: {
    num: 121,
    name: "Starmie",
    types: ["Water", "Psychic"],
    baseStats: { hp: 60, atk: 75, def: 85, spa: 100, spd: 85, spe: 115 },
    abilities: { 0: "Illuminate" },
  },
  marowak: {
    num: 105,
    name: "Marowak",
    types: ["Ground"],
    baseStats: { hp: 60, atk: 80, def: 110, spa: 50, spd: 80, spe: 45 },
    abilities: { 0: "Rock Head" },
  },
};
""".lstrip(),
        encoding="utf-8",
    )


def _metadata(
    name: str,
    *,
    category: str = "Physical",
    base_power: int = 50,
    accuracy: int | bool = 100,
    status: str | None = None,
    volatile_status: str | None = None,
    boosts: dict | None = None,
    secondary: dict | None = None,
    recharge: bool = False,
    charge: bool = False,
    high_crit: bool = False,
    move_type: str = "Normal",
) -> dict:
    return {
        "requested_name": name,
        "id": name.lower().replace(" ", ""),
        "name": name,
        "type": move_type,
        "category": category,
        "base_power": base_power,
        "accuracy": accuracy,
        "status": status,
        "volatile_status": volatile_status,
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
