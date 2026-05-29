from __future__ import annotations

from pathlib import Path
import random
import unittest
from unittest import mock

from pokerena.custom_bot import CustomBotPlan
from pokerena.eval_tracker import (
    EVAL_BOT_SPECS,
    _eval_choice,
    _refresh_summary,
    _resolve_bot_specs,
    _winner_from_finished_payload,
)


class EvalTrackerTest(unittest.TestCase):
    def test_winner_falls_back_to_public_battle_log(self) -> None:
        winner = _winner_from_finished_payload(
            {},
            [
                "|turn|9",
                "|faint|p2a: Snorlax",
                "|win|Gen1MaxDamageBot",
            ],
        )

        self.assertEqual(winner, "Gen1MaxDamageBot")

    def test_winner_prefers_finished_payload(self) -> None:
        winner = _winner_from_finished_payload(
            {"winner": "Gen1CustomBot"},
            ["|win|Gen1MaxDamageBot"],
        )

        self.assertEqual(winner, "Gen1CustomBot")

    def test_resolves_distinct_pool_bots(self) -> None:
        specs = _resolve_bot_specs(
            ["custom-bot", "custom-bot-cube", "max-damage-bot", "random-bot", "custom-bot"]
        )

        self.assertEqual(
            [spec.label for spec in specs],
            ["Gen1CustomBot", "Gen1CustomBotCube", "Gen1MaxDamageBot", "Gen1RandomBot"],
        )

    def test_custom_eval_variant_passes_selection_strategy(self) -> None:
        plan = CustomBotPlan(decision="move 1", notes="scored", actions=[])
        with (
            mock.patch("pokerena.eval_tracker.asdict", return_value={"request": {}}),
            mock.patch("pokerena.eval_tracker.build_custom_bot_plan", return_value=plan) as build,
        ):
            choice, fallback_reason, error = _eval_choice(
                spec=EVAL_BOT_SPECS["custom-bot-linear"],
                context=object(),
                capture_payload={},
                rng=random.Random(1),
                project_root=Path.cwd(),
            )

        self.assertEqual(choice, "move 1")
        self.assertIsNone(fallback_reason)
        self.assertIsNone(error)
        self.assertEqual(build.call_args.kwargs["selection_strategy"], "weighted-linear")

    def test_refresh_summary_tracks_pair_results(self) -> None:
        state = {
            "game_results": [
                {
                    "pair_key": "Gen1CustomBot vs Gen1RandomBot",
                    "p1": "Gen1CustomBot",
                    "p2": "Gen1RandomBot",
                    "winner": "Gen1CustomBot",
                    "winner_side": "p1",
                    "turns": 20,
                    "errors": [],
                    "fallback_reasons": {},
                },
                {
                    "pair_key": "Gen1CustomBot vs Gen1RandomBot",
                    "p1": "Gen1RandomBot",
                    "p2": "Gen1CustomBot",
                    "winner": "Gen1RandomBot",
                    "winner_side": "p1",
                    "turns": 30,
                    "errors": [],
                    "fallback_reasons": {},
                },
            ]
        }

        _refresh_summary(state)

        pair = state["pair_results"]["Gen1CustomBot vs Gen1RandomBot"]
        self.assertEqual(pair["games"], 2)
        self.assertEqual(pair["wins"], {"Gen1CustomBot": 1, "Gen1RandomBot": 1})
        self.assertEqual(pair["win_rates"], {"Gen1CustomBot": 50.0, "Gen1RandomBot": 50.0})
        self.assertEqual(pair["mean_turns"], 25)


if __name__ == "__main__":
    unittest.main()
