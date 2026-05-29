from __future__ import annotations

import random
import unittest

from pokerena.random_bot import build_random_bot_decision


class RandomBotTest(unittest.TestCase):
    def test_move_turn_chooses_random_enabled_move_when_not_switching(self) -> None:
        decision = build_random_bot_decision(
            _context(),
            rng=random.Random(4),
            switch_chance=0.0,
        )

        self.assertIn(decision.decision, {"move 1", "move 2"})
        self.assertIn("random enabled move", decision.notes)

    def test_move_turn_can_make_voluntary_random_switch(self) -> None:
        decision = build_random_bot_decision(
            _context(),
            rng=random.Random(4),
            switch_chance=1.0,
        )

        self.assertIn(decision.decision, {"switch 2", "switch 3"})
        self.assertIn("voluntary switch", decision.notes)

    def test_disabled_moves_are_not_selected_on_move_turns(self) -> None:
        decision = build_random_bot_decision(
            _context(
                moves=[
                    {"move": "Recover", "id": "recover", "disabled": True},
                    {"move": "Surf", "id": "surf", "disabled": False},
                ],
            ),
            rng=random.Random(9),
            switch_chance=0.0,
        )

        self.assertEqual(decision.decision, "move 2")


def _context(*, moves: list[dict] | None = None) -> dict:
    moves = moves or [
        {"move": "Body Slam", "id": "bodyslam", "disabled": False},
        {"move": "Surf", "id": "surf", "disabled": False},
    ]
    request = {
        "active": [{"moves": moves}],
        "side": {
            "id": "p1",
            "pokemon": [
                {"ident": "p1: Lapras", "details": "Lapras, L80", "condition": "100/100", "active": True},
                {"ident": "p1: Snorlax", "details": "Snorlax, L80", "condition": "100/100", "active": False},
                {"ident": "p1: Starmie", "details": "Starmie, L80", "condition": "100/100", "active": False},
            ],
        },
    }
    return {
        "schema_version": "pokerena.turn-context.v1",
        "battle_id": "battle-gen1",
        "agent_id": "random",
        "provider": "pokerena-random-bot",
        "player_slot": "p1",
        "context_token": "battle-gen1:p1:1:1",
        "format_name": "gen1randombattle",
        "turn_number": 1,
        "phase": "turn",
        "request_kind": "move",
        "rqid": "1",
        "request_sequence": 1,
        "decision_attempt": 1,
        "signals": {"decision_required": True},
        "legal_action_hints": ["move 1", "move 2", "switch 2", "switch 3"],
        "request": request,
        "side": request["side"],
        "active": request["active"],
        "recent_public_events": [],
        "last_error": None,
    }


if __name__ == "__main__":
    unittest.main()
