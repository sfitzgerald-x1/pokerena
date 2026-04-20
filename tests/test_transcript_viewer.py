from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import tempfile
import textwrap
import threading
import unittest
from unittest import mock
from urllib import request as urllib_request
from urllib.error import HTTPError

from pokerena.config import load_agents_config
from pokerena.transcript import (
    TranscriptEntry,
    TranscriptTraceEvent,
    default_transcript_path,
    record_transcript_entry,
    upsert_battle_summary_entry,
    update_transcript_metadata,
)
from pokerena.transcript_viewer import _build_handler


class TranscriptViewerTest(unittest.TestCase):
    def test_viewer_endpoints_return_battle_summaries_and_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: claude-bot
                        enabled: true
                        provider: claude
                        player_slot: p1
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
            runtime_root = root / ".runtime"
            transcript_path = default_transcript_path(runtime_root, agent, "battle-gen3randombattle-1")
            update_transcript_metadata(
                transcript_path,
                agent=agent,
                battle_id="battle-gen3randombattle-1",
                format_name="gen3randombattle",
                challenger="human",
                finished=False,
            )
            record_transcript_entry(
                transcript_path,
                agent=agent,
                battle_id="battle-gen3randombattle-1",
                format_name="gen3randombattle",
                challenger="human",
                winner=None,
                finished=False,
                entry=TranscriptEntry(
                    turn_number=1,
                    request_sequence=1,
                    request_kind="move",
                    rqid="7",
                    decision_attempt=1,
                    prompt_text="Prompt text",
                    recent_public_events=["|turn|1", "|move|p1a: A|Surf|p2a: B"],
                    decision="move 1",
                    raw_output='{"decision":"move 1"}',
                    notes="safe move",
                    fallback_used=False,
                    fallback_reason=None,
                    error=None,
                    turn_state="accepted",
                    submission_state="accepted",
                    submission_detail="Battle progress confirmed the submitted choice.",
                    selected_action="move 1",
                    selected_action_label='Used "Thunderbolt"',
                    selected_action_source="agent",
                    timeout_seconds=120,
                    trace_events=[
                        TranscriptTraceEvent(kind="status", message="Prompt built.", created_at="2026-04-19T00:00:00Z"),
                        TranscriptTraceEvent(
                            kind="status",
                            message="Calculated damage ranges.",
                            created_at="2026-04-19T00:00:00Z",
                            actor_kind="helper",
                            actor_name="calc-worker",
                        ),
                        TranscriptTraceEvent(
                            kind="agent",
                            message='{"decision":"move 1"}',
                            created_at="2026-04-19T00:00:00Z",
                            actor_kind="agent",
                        ),
                    ],
                    decision_latency_ms=950,
                    usage={"provider": "claude", "total_tokens": 123},
                    submitted_at="2026-04-19T00:00:00Z",
                    validated_at="2026-04-19T00:00:01Z",
                    created_at="2026-04-19T00:00:00Z",
                ),
            )
            update_transcript_metadata(
                transcript_path,
                agent=agent,
                battle_id="battle-gen3randombattle-1",
                winner="ClaudeLocalBot",
                finished=True,
            )
            upsert_battle_summary_entry(transcript_path)

            server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler(runtime_root))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                with urllib_request.urlopen(f"{base_url}/", timeout=2) as response:
                    html = response.read().decode("utf-8")
                with urllib_request.urlopen(f"{base_url}/api/battles", timeout=2) as response:
                    summaries = json.loads(response.read().decode("utf-8"))
                with urllib_request.urlopen(
                    f"{base_url}/api/battles/claude-bot/battle-gen3randombattle-1",
                    timeout=2,
                ) as response:
                    battle = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            self.assertIn("Battle Sessions", html)
            self.assertEqual(len(summaries["battles"]), 1)
            self.assertTrue(summaries["battles"][0]["finished"])
            self.assertEqual(summaries["battles"][0]["winner"], "ClaudeLocalBot")
            self.assertEqual(summaries["battles"][0]["helper_summary"]["count"], 1)
            self.assertEqual(battle["battle_id"], "battle-gen3randombattle-1")
            self.assertEqual(battle["entries"][0]["decision"], "move 1")
            self.assertEqual(battle["entries"][0]["submission_state"], "accepted")
            self.assertEqual(battle["entries"][0]["turn_state"], "accepted")
            self.assertEqual(battle["entries"][0]["selected_action_label"], 'Used "Thunderbolt"')
            self.assertEqual(battle["entries"][0]["selected_action_source"], "agent")
            self.assertEqual(battle["helper_summary"]["count"], 1)
            self.assertEqual(battle["entries"][-1]["entry_kind"], "summary")
            self.assertEqual(battle["entries"][-1]["summary"]["result"], "ClaudeLocalBot")
            self.assertEqual(battle["entries"][-1]["summary"]["total_turns"], 1)
            self.assertEqual(battle["entries"][-1]["summary"]["average_decision_latency_ms"], 950)
            self.assertEqual(battle["entries"][-1]["summary"]["usage"]["total_tokens"], 123)

    def test_viewer_endpoints_tolerate_older_transcripts_without_action_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: claude-bot
                        enabled: true
                        provider: claude
                        player_slot: p1
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
            runtime_root = root / ".runtime"
            transcript_path = default_transcript_path(runtime_root, agent, "battle-gen3randombattle-legacy")
            record_transcript_entry(
                transcript_path,
                agent=agent,
                battle_id="battle-gen3randombattle-legacy",
                format_name="gen3randombattle",
                challenger="human",
                winner=None,
                finished=False,
                entry=TranscriptEntry(
                    turn_number=1,
                    request_sequence=1,
                    request_kind="move",
                    rqid="3",
                    decision_attempt=1,
                    prompt_text="Prompt text",
                    recent_public_events=["|turn|1"],
                    selected_action="move 1",
                    selected_action_source="agent",
                ),
            )
            payload = json.loads(transcript_path.read_text(encoding="utf-8"))
            payload["entries"][0].pop("selected_action_label", None)
            transcript_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler(runtime_root))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                with urllib_request.urlopen(
                    f"{base_url}/api/battles/claude-bot/battle-gen3randombattle-legacy",
                    timeout=2,
                ) as response:
                    battle = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            self.assertEqual(battle["entries"][0]["selected_action"], "move 1")
            self.assertNotIn("selected_action_label", battle["entries"][0])

    def test_delete_endpoint_only_allows_finished_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: claude-bot
                        enabled: true
                        provider: claude
                        player_slot: p1
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
            runtime_root = root / ".runtime"

            finished_path = default_transcript_path(runtime_root, agent, "battle-finished")
            update_transcript_metadata(
                finished_path,
                agent=agent,
                battle_id="battle-finished",
                format_name="gen3randombattle",
                challenger="human",
                winner="ClaudeLocalBot",
                finished=False,
            )
            live_path = default_transcript_path(runtime_root, agent, "battle-live")
            update_transcript_metadata(
                live_path,
                agent=agent,
                battle_id="battle-live",
                format_name="gen3randombattle",
                challenger="human",
                finished=False,
            )

            server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler(runtime_root))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                with mock.patch(
                    "pokerena.transcript.send2trash",
                    side_effect=lambda path: shutil.rmtree(path),
                ):
                    request = urllib_request.Request(
                        f"{base_url}/api/battles/claude-bot/battle-finished",
                        method="DELETE",
                    )
                    with urllib_request.urlopen(request, timeout=2) as response:
                        deleted = json.loads(response.read().decode("utf-8"))

                self.assertTrue(deleted["ok"])
                with urllib_request.urlopen(f"{base_url}/api/battles", timeout=2) as response:
                    summaries = json.loads(response.read().decode("utf-8"))
                self.assertEqual([battle["battle_id"] for battle in summaries["battles"]], ["battle-live"])

                with self.assertRaises(HTTPError) as live_error:
                    urllib_request.urlopen(
                        urllib_request.Request(
                            f"{base_url}/api/battles/claude-bot/battle-live",
                            method="DELETE",
                        ),
                        timeout=2,
                    )
                self.assertEqual(live_error.exception.code, 409)
                live_error.exception.close()

                with self.assertRaises(HTTPError) as missing_error:
                    urllib_request.urlopen(
                        urllib_request.Request(
                            f"{base_url}/api/battles/claude-bot/battle-missing",
                            method="DELETE",
                        ),
                        timeout=2,
                    )
                self.assertEqual(missing_error.exception.code, 404)
                missing_error.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_stop_endpoint_marks_live_session_for_forfeit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            (root / "config" / "agents.yaml").write_text(
                textwrap.dedent(
                    """
                    agents:
                      - id: claude-bot
                        enabled: true
                        provider: claude
                        player_slot: p1
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
            runtime_root = root / ".runtime"
            transcript_path = default_transcript_path(runtime_root, agent, "battle-live")
            update_transcript_metadata(
                transcript_path,
                agent=agent,
                battle_id="battle-live",
                format_name="gen3randombattle",
                challenger="human",
                finished=False,
            )

            server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler(runtime_root))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                request = urllib_request.Request(
                    f"{base_url}/api/battles/claude-bot/battle-live/stop",
                    method="POST",
                )
                with urllib_request.urlopen(request, timeout=2) as response:
                    stopped = json.loads(response.read().decode("utf-8"))
                with urllib_request.urlopen(f"{base_url}/api/battles", timeout=2) as response:
                    summaries = json.loads(response.read().decode("utf-8"))
                with urllib_request.urlopen(
                    f"{base_url}/api/battles/claude-bot/battle-live",
                    timeout=2,
                ) as response:
                    battle = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            self.assertTrue(stopped["ok"])
            self.assertIsInstance(stopped["requested_at"], str)
            self.assertFalse(summaries["battles"][0]["finished"])
            self.assertEqual(summaries["battles"][0]["stop_requested_at"], stopped["requested_at"])
            self.assertEqual(battle["stop_requested_at"], stopped["requested_at"])
