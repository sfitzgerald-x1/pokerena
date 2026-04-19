from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from .config import ServerConfig
from .transcript import (
    BattleSessionDeleteConflictError,
    BattleSessionStopConflictError,
    delete_battle_session,
    load_battle_transcript,
    request_battle_stop,
    list_transcript_summaries,
)


def transcript_viewer_url(server_config: ServerConfig) -> str:
    host = _format_http_host(server_config.transcript_viewer.bind_address)
    return f"http://{host}:{server_config.transcript_viewer.port}"


def serve_transcript_viewer(server_config: ServerConfig) -> None:
    runtime_root = server_config.project_root / ".runtime"
    handler = _build_handler(runtime_root, allowed_origin=transcript_viewer_url(server_config))
    httpd = ThreadingHTTPServer(
        (server_config.transcript_viewer.bind_address, server_config.transcript_viewer.port),
        handler,
    )
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


def _build_handler(runtime_root: Path, allowed_origin: Optional[str] = None):
    class TranscriptViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self._send_html(_INDEX_HTML)
                return
            if self.path == "/healthz":
                self._send_json({"ok": True})
                return
            if self.path == "/api/battles":
                self._send_json({"battles": list_transcript_summaries(runtime_root)})
                return
            if self.path.startswith("/api/battles/"):
                battle = self._battle_payload()
                if battle is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "Battle transcript not found.")
                    return
                self._send_json(battle)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found.")

        def do_DELETE(self) -> None:  # noqa: N802
            if not self._origin_allowed():
                self.send_error(HTTPStatus.FORBIDDEN, "Origin not allowed.")
                return
            identifiers = self._battle_identifiers()
            if identifiers is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Battle transcript not found.")
                return
            agent_id, battle_id = identifiers
            try:
                delete_battle_session(runtime_root, agent_id, battle_id)
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND, "Battle transcript not found.")
                return
            except BattleSessionDeleteConflictError as error:
                self._send_json({"ok": False, "error": str(error)}, status=HTTPStatus.CONFLICT)
                return
            self._send_json({"ok": True, "agent_id": agent_id, "battle_id": battle_id})

        def do_POST(self) -> None:  # noqa: N802
            if not self._origin_allowed():
                self.send_error(HTTPStatus.FORBIDDEN, "Origin not allowed.")
                return
            identifiers = self._battle_identifiers(action="stop")
            if identifiers is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Battle transcript not found.")
                return
            agent_id, battle_id = identifiers
            try:
                payload = request_battle_stop(runtime_root, agent_id, battle_id)
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND, "Battle transcript not found.")
                return
            except BattleSessionStopConflictError as error:
                self._send_json({"ok": False, "error": str(error)}, status=HTTPStatus.CONFLICT)
                return
            self._send_json(
                {
                    "ok": True,
                    "agent_id": agent_id,
                    "battle_id": battle_id,
                    "requested_at": payload.get("requested_at"),
                }
            )

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _origin_allowed(self) -> bool:
            if not allowed_origin:
                return True
            origin = self.headers.get("Origin")
            if not origin:
                return True
            return origin == allowed_origin

        def _battle_identifiers(self, *, action: Optional[str] = None) -> Optional[tuple[str, str]]:
            parts = [unquote(part) for part in self.path.split("/")[3:] if part]
            if action is None:
                if len(parts) != 2:
                    return None
                return parts[0], parts[1]
            if len(parts) != 3 or parts[2] != action:
                return None
            return parts[0], parts[1]

        def _battle_payload(self) -> Optional[dict]:
            identifiers = self._battle_identifiers()
            if identifiers is None:
                return None
            agent_id, battle_id = identifiers
            return load_battle_transcript(runtime_root, agent_id, battle_id)

        def _send_html(self, content: str) -> None:
            body = content.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return TranscriptViewerHandler


def _format_http_host(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pokerena Battle Sessions</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4efe7;
      --panel: #fffaf3;
      --panel-strong: #fff5eb;
      --ink: #2b2016;
      --muted: #7b6553;
      --accent: #cc6b2c;
      --accent-soft: #f2d8c4;
      --border: #d8c6b6;
      --shadow: 0 14px 40px rgba(83, 48, 22, 0.08);
      --ok: #3f7d4a;
      --warn: #9a5b21;
      --bad: #a63731;
      --thinking: #356f96;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      background: radial-gradient(circle at top, #fff6eb 0%, var(--bg) 55%, #efe4d7 100%);
      color: var(--ink);
    }
    button {
      font: inherit;
      color: inherit;
      cursor: pointer;
    }
    .layout {
      display: grid;
      grid-template-columns: 340px 1fr;
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--border);
      background: rgba(255, 250, 243, 0.92);
      backdrop-filter: blur(12px);
      padding: 24px 18px;
      overflow: auto;
    }
    .sidebar h1 {
      margin: 0 0 8px;
      font-size: 28px;
    }
    .sidebar p {
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.45;
    }
    .battle-list {
      display: grid;
      gap: 12px;
    }
    .battle-card {
      display: grid;
      gap: 8px;
      padding: 14px;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: white;
      box-shadow: var(--shadow);
    }
    .battle-card.active {
      border-color: var(--accent);
      background: #fff1e5;
    }
    .battle-top {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: flex-start;
    }
    .battle-card strong {
      display: block;
      margin-bottom: 4px;
      font-size: 16px;
    }
    .battle-meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .battle-actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .chip {
      display: inline-block;
      padding: 4px 9px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #7d431f;
      font-size: 12px;
      font-weight: 600;
    }
    .chip.ok {
      background: #d6efd9;
      color: #24532d;
    }
    .chip.warn {
      background: #f5dfcf;
      color: #7a471f;
    }
    .chip.bad {
      background: #f5d0ce;
      color: #7a2924;
    }
    .ghost-button {
      border: 1px solid var(--border);
      border-radius: 999px;
      background: rgba(255,255,255,0.88);
      padding: 7px 12px;
    }
    .danger-button {
      border-color: #d7a8a4;
      background: #fff2f1;
      color: #8d312d;
    }
    .main {
      padding: 28px 36px 48px;
      overflow: auto;
    }
    .hero {
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
    }
    .hero h2 {
      margin: 0 0 6px;
      font-size: 34px;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
    }
    .cards {
      display: grid;
      gap: 18px;
      max-width: 1100px;
    }
    .turn-card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 18px 20px;
      box-shadow: var(--shadow);
    }
    .turn-card.live {
      background: linear-gradient(180deg, var(--panel-strong) 0%, var(--panel) 100%);
    }
    .turn-header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 14px;
    }
    .turn-title {
      margin: 0 0 6px;
      font-size: 20px;
    }
    .turn-meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .header-actions {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .state-chip {
      color: white;
    }
    .state-thinking { background: var(--thinking); }
    .state-submitted { background: #8b6f4b; }
    .state-accepted { background: var(--ok); }
    .state-rejected { background: var(--bad); }
    .state-timed-out { background: var(--warn); }
    .state-error { background: var(--bad); }
    .state-stopped { background: var(--bad); }
    .chip-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }
    .helper-strip {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }
    .summary {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
      margin-bottom: 12px;
    }
    .decision-box,
    .summary-grid {
      border: 1px solid var(--border);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.82);
      padding: 12px 14px;
      margin-bottom: 12px;
    }
    .decision-box strong,
    .summary-grid strong {
      display: block;
      margin-bottom: 6px;
      font-size: 14px;
    }
    .decision-line {
      font-family: "SFMono-Regular", Menlo, Monaco, monospace;
      font-size: 14px;
      color: #2f241b;
      margin-bottom: 6px;
    }
    .summary-grid {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    }
    .summary-item {
      min-height: 56px;
    }
    .trace-controls {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 8px;
    }
    .trace-controls strong {
      font-size: 14px;
    }
    .trace-shell {
      border: 1px solid #eadccd;
      border-radius: 16px;
      background: rgba(255,255,255,0.78);
      overflow: hidden;
    }
    .trace-shell.collapsed .trace-log {
      display: none;
    }
    .trace-log {
      margin: 0;
      padding: 14px;
      max-height: 240px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Menlo, Monaco, monospace;
      font-size: 13px;
      line-height: 1.5;
      color: #2f241b;
    }
    .empty {
      padding: 28px;
      border: 1px dashed var(--border);
      border-radius: 18px;
      color: var(--muted);
      background: rgba(255, 250, 243, 0.7);
      max-width: 720px;
    }
    .modal.hidden {
      display: none;
    }
    .modal {
      position: fixed;
      inset: 0;
      background: rgba(43, 32, 22, 0.45);
      display: grid;
      place-items: center;
      padding: 20px;
    }
    .modal-panel {
      width: min(980px, 100%);
      max-height: 90vh;
      overflow: auto;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 20px;
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 14px;
    }
    .modal-header h3 {
      margin: 0;
      font-size: 24px;
    }
    .modal-meta {
      color: var(--muted);
      margin-bottom: 12px;
      line-height: 1.5;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Menlo, Monaco, monospace;
      font-size: 13px;
      line-height: 1.45;
      color: #2f241b;
      background: rgba(255,255,255,0.78);
      border: 1px solid #eadccd;
      border-radius: 14px;
      padding: 12px;
    }
    .modal-section {
      display: grid;
      gap: 8px;
      margin-top: 14px;
    }
    .modal-section strong {
      font-size: 14px;
    }
    @media (max-width: 960px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--border); max-height: 40vh; }
      .main { padding: 20px; }
      .turn-header, .hero { flex-direction: column; }
      .header-actions { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside id="sidebar" class="sidebar">
      <h1>Battle Sessions</h1>
      <p>Browse finished and live local sessions, keep your scroll position while traces update, and inspect prompts only when you need them.</p>
      <div id="battle-list" class="battle-list"></div>
    </aside>
    <main id="main" class="main">
      <div class="hero">
        <div>
          <h2 id="battle-title">No session selected</h2>
          <p id="battle-subtitle">Open a local battle and the session transcript will appear here.</p>
        </div>
        <div id="hero-actions" class="battle-actions"></div>
      </div>
      <div id="messages" class="cards"></div>
    </main>
  </div>

  <div id="prompt-modal" class="modal hidden" role="dialog" aria-modal="true">
    <div class="modal-panel">
      <div class="modal-header">
        <h3 id="prompt-title">Prompt</h3>
        <button id="prompt-close" class="ghost-button" type="button">Close</button>
      </div>
      <div id="prompt-meta" class="modal-meta"></div>
      <div class="modal-section">
        <strong>Rendered prompt</strong>
        <pre id="prompt-body"></pre>
      </div>
      <div class="modal-section">
        <strong>Recent public battle history</strong>
        <pre id="prompt-history"></pre>
      </div>
    </div>
  </div>

  <script>
    let selectedBattle = null;
    let selectedAgent = null;
    const expandedTraces = new Set();
    const tracePaneState = new Map();

    function battleKey(agentId, battleId) {
      return `${agentId}/${battleId}`;
    }

    function turnKey(agentId, battleId, requestSequence) {
      return `${battleKey(agentId, battleId)}:${requestSequence}`;
    }

    function readSelectionFromHash() {
      const hash = window.location.hash.replace(/^#/, "");
      if (!hash) return;
      const [agentId, battleId] = hash.split("/", 2);
      if (agentId && battleId) {
        selectedAgent = decodeURIComponent(agentId);
        selectedBattle = decodeURIComponent(battleId);
      }
    }

    function updateHash(agentId, battleId) {
      window.location.hash = `${encodeURIComponent(agentId)}/${encodeURIComponent(battleId)}`;
    }

    async function fetchBattles() {
      const response = await fetch("/api/battles", {cache: "no-store"});
      return await response.json();
    }

    async function fetchBattle(agentId, battleId) {
      const response = await fetch(`/api/battles/${encodeURIComponent(agentId)}/${encodeURIComponent(battleId)}`, {cache: "no-store"});
      if (response.status === 404) return null;
      return await response.json();
    }

    async function deleteBattle(agentId, battleId) {
      const response = await fetch(`/api/battles/${encodeURIComponent(agentId)}/${encodeURIComponent(battleId)}`, {
        method: "DELETE",
      });
      if (response.status === 409) {
        const payload = await response.json();
        throw new Error(payload.error || "Battle session cannot be deleted while live.");
      }
      if (!response.ok) {
        throw new Error(`Delete failed with status ${response.status}.`);
      }
      return await response.json();
    }

    async function stopBattle(agentId, battleId) {
      const response = await fetch(`/api/battles/${encodeURIComponent(agentId)}/${encodeURIComponent(battleId)}/stop`, {
        method: "POST",
      });
      if (response.status === 409) {
        const payload = await response.json();
        throw new Error(payload.error || "Battle session cannot be stopped.");
      }
      if (!response.ok) {
        throw new Error(`Stop failed with status ${response.status}.`);
      }
      return await response.json();
    }

    function statusLabel(battle) {
      if (battle.finished) {
        return `finished${battle.winner ? ` · winner ${battle.winner}` : ""}`;
      }
      if (battle.stop_requested_at) {
        return "stopping";
      }
      return "live";
    }

    function formatStatus(battle) {
      return statusLabel(battle);
    }

    function formatAbsoluteTimestamp(value) {
      if (!value) return "unknown time";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return "unknown time";
      return new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(date);
    }

    function formatRelativeTimestamp(value) {
      if (!value) return "unknown age";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return "unknown age";
      const diffMs = date.getTime() - Date.now();
      const rtf = new Intl.RelativeTimeFormat(undefined, {numeric: "auto"});
      const units = [
        [60 * 1000, "second"],
        [60 * 60 * 1000, "minute"],
        [24 * 60 * 60 * 1000, "hour"],
        [7 * 24 * 60 * 60 * 1000, "day"],
      ];
      for (const [threshold, unit] of units) {
        if (Math.abs(diffMs) < threshold) {
          const divisor = unit === "second" ? 1000 : unit === "minute" ? 60 * 1000 : unit === "hour" ? 60 * 60 * 1000 : 24 * 60 * 60 * 1000;
          return rtf.format(Math.round(diffMs / divisor), unit);
        }
      }
      return rtf.format(Math.round(diffMs / (7 * 24 * 60 * 60 * 1000)), "week");
    }

    function helperSummaryText(helperSummary) {
      if (!helperSummary || !Array.isArray(helperSummary.actors) || !helperSummary.actors.length) {
        return "";
      }
      return helperSummary.actors.slice(0, 3).map(actor => `${actor.name} ×${actor.count}`).join(" · ");
    }

    function renderBattleList(battles) {
      const list = document.getElementById("battle-list");
      list.innerHTML = "";
      if (!battles.length) {
        list.innerHTML = '<div class="empty">No sessions yet. Start a battle and the local transcript will appear here.</div>';
        return;
      }
      const selectedExists = battles.some(battle => battle.agent_id === selectedAgent && battle.battle_id === selectedBattle);
      if ((!selectedAgent || !selectedBattle || !selectedExists) && battles[0]) {
        selectedAgent = battles[0].agent_id;
        selectedBattle = battles[0].battle_id;
        updateHash(selectedAgent, selectedBattle);
      }
      for (const battle of battles) {
        const card = document.createElement("section");
        card.className = "battle-card";
        if (battle.agent_id === selectedAgent && battle.battle_id === selectedBattle) {
          card.classList.add("active");
        }
        const helperSummary = helperSummaryText(battle.helper_summary);
        card.innerHTML = `
          <div class="battle-top">
            <div>
              <strong>${escapeHtml(battle.battle_id)}</strong>
              <div class="battle-meta">${escapeHtml(battle.format_name || "Unknown format")}<br>${escapeHtml(battle.agent_id)} vs ${escapeHtml(battle.challenger || "local player")}<br>${escapeHtml(formatAbsoluteTimestamp(battle.updated_at))} · ${escapeHtml(formatRelativeTimestamp(battle.updated_at))}</div>
            </div>
            <div class="battle-actions">
              <span class="chip ${battle.finished ? "ok" : "warn"}">${escapeHtml(formatStatus(battle))}</span>
              ${battle.finished
                ? '<button class="ghost-button danger-button delete-session" type="button">Delete</button>'
                : battle.transport === "showdown-client"
                  ? `<button class="ghost-button danger-button stop-session" type="button"${battle.stop_requested_at ? " disabled" : ""}>${battle.stop_requested_at ? "Stopping…" : "Stop agent"}</button>`
                  : ""}
            </div>
          </div>
          ${helperSummary ? `<div class="battle-meta">Helpers · ${escapeHtml(helperSummary)}</div>` : ""}
        `;
        card.addEventListener("click", () => {
          selectedAgent = battle.agent_id;
          selectedBattle = battle.battle_id;
          updateHash(selectedAgent, selectedBattle);
          refresh();
        });
        const deleteButton = card.querySelector(".delete-session");
        if (deleteButton) {
          deleteButton.addEventListener("click", async (event) => {
            event.stopPropagation();
            const confirmed = window.confirm(`Move session ${battle.battle_id} to Trash?`);
            if (!confirmed) return;
            try {
              await deleteBattle(battle.agent_id, battle.battle_id);
              if (selectedAgent === battle.agent_id && selectedBattle === battle.battle_id) {
                selectedAgent = null;
                selectedBattle = null;
                window.location.hash = "";
              }
              await refresh();
            } catch (error) {
              window.alert(error instanceof Error ? error.message : String(error));
            }
          });
        }
        const stopButton = card.querySelector(".stop-session");
        if (stopButton && !battle.stop_requested_at) {
          stopButton.addEventListener("click", async (event) => {
            event.stopPropagation();
            const confirmed = window.confirm(`Forfeit ${battle.battle_id} and stop the agent from making any more decisions in this battle?`);
            if (!confirmed) return;
            try {
              await stopBattle(battle.agent_id, battle.battle_id);
              await refresh();
            } catch (error) {
              window.alert(error instanceof Error ? error.message : String(error));
            }
          });
        }
        list.appendChild(card);
      }
    }

    function normalizeTraceEvent(event) {
      return {
        kind: event && event.kind ? event.kind : "status",
        message: event && event.message ? String(event.message) : "",
        created_at: event && event.created_at ? String(event.created_at) : "",
        actor_kind: event && event.actor_kind ? String(event.actor_kind) : (event && event.kind === "agent" ? "agent" : "pokerena"),
        actor_name: event && event.actor_name ? String(event.actor_name) : null,
      };
    }

    function normalizeEntry(entry) {
      return {
        ...entry,
        entry_kind: entry.entry_kind || "turn",
        decision: entry.decision || null,
        raw_output: entry.raw_output || "",
        notes: entry.notes || "",
        fallback_reason: entry.fallback_reason || null,
        error: entry.error || null,
        submission_state: entry.submission_state || "pending",
        submission_detail: entry.submission_detail || null,
        turn_state: entry.turn_state || "thinking",
        selected_action: entry.selected_action || entry.decision || null,
        selected_action_label: entry.selected_action_label || null,
        selected_action_source: entry.selected_action_source || (entry.decision ? "agent" : null),
        timeout_seconds: entry.timeout_seconds || null,
        trace_events: Array.isArray(entry.trace_events) ? entry.trace_events.map(normalizeTraceEvent) : [],
        recent_public_events: Array.isArray(entry.recent_public_events) ? entry.recent_public_events : [],
        decision_latency_ms: Number.isInteger(entry.decision_latency_ms) ? entry.decision_latency_ms : null,
        usage: entry.usage && typeof entry.usage === "object" ? entry.usage : null,
        summary: entry.summary && typeof entry.summary === "object" ? entry.summary : null,
        created_at: entry.created_at || "",
        submitted_at: entry.submitted_at || null,
        timed_out_at: entry.timed_out_at || null,
        validated_at: entry.validated_at || null,
      };
    }

    function groupEntries(entries) {
      const grouped = new Map();
      for (const rawEntry of entries) {
        const entry = normalizeEntry(rawEntry);
        const key = entry.entry_kind === "summary" ? `summary-${entry.request_sequence}` : `turn-${entry.request_sequence}`;
        if (!grouped.has(key)) {
          grouped.set(key, {requestSequence: entry.request_sequence, entryKind: entry.entry_kind, attempts: []});
        }
        grouped.get(key).attempts.push(entry);
      }
      const groups = Array.from(grouped.values());
      for (const group of groups) {
        group.attempts.sort((left, right) => (left.decision_attempt || 0) - (right.decision_attempt || 0));
        group.latest = group.attempts[group.attempts.length - 1];
      }
      groups.sort((left, right) => {
        const leftStamp = left.latest.created_at || "";
        const rightStamp = right.latest.created_at || "";
        if (leftStamp === rightStamp) {
          return right.requestSequence - left.requestSequence;
        }
        return rightStamp.localeCompare(leftStamp);
      });
      return groups;
    }

    function displayTurnState(entry) {
      if (entry.turn_state === "thinking" && entry.fallback_reason === "timeout") {
        return "timed out";
      }
      return String(entry.turn_state || "thinking").replace(/[-_]/g, " ");
    }

    function summarizeHelpers(group) {
      const counts = new Map();
      for (const attempt of group.attempts) {
        for (const event of attempt.trace_events) {
          if (event.actor_kind !== "helper") continue;
          const name = event.actor_name || "helper";
          counts.set(name, (counts.get(name) || 0) + 1);
        }
      }
      return Array.from(counts.entries()).map(([name, count]) => ({name, count}));
    }

    function traceText(group) {
      const lines = [];
      for (const attempt of group.attempts) {
        if (group.entryKind !== "summary" && group.attempts.length > 1) {
          lines.push(`Attempt ${attempt.decision_attempt}`);
        }
        if (attempt.trace_events.length) {
          for (const event of attempt.trace_events) {
            const prefix = event.actor_kind === "helper"
              ? `Helper${event.actor_name ? ` (${event.actor_name})` : ""}`
              : event.actor_kind === "agent"
                ? "Agent"
                : "Pokerena";
            lines.push(`${prefix}: ${event.message}`);
          }
        } else if (attempt.raw_output) {
          lines.push(`Agent: ${attempt.raw_output}`);
        } else if (group.entryKind !== "summary") {
          lines.push("Pokerena: Waiting for agent output...");
        }
      }
      return lines.join("\\n");
    }

    function formatDurationMs(value) {
      if (!Number.isInteger(value)) return "n/a";
      if (value < 1000) return `${value}ms`;
      return `${(value / 1000).toFixed(1)}s`;
    }

    function formatUsage(usage) {
      if (!usage || typeof usage.total_tokens !== "number") {
        return "unavailable";
      }
      return `${usage.total_tokens.toLocaleString()} total`;
    }

    function formatUsdCost(costEstimate) {
      if (!costEstimate || typeof costEstimate.total_usd !== "string") {
        return "unavailable";
      }
      return `$${costEstimate.total_usd}`;
    }

    function captureTracePaneState() {
      document.querySelectorAll(".trace-log").forEach(node => {
        const key = node.dataset.traceKey;
        if (!key) return;
        const atBottom = node.scrollTop + node.clientHeight >= node.scrollHeight - 12;
        tracePaneState.set(key, {scrollTop: node.scrollTop, atBottom});
      });
    }

    function restoreTracePaneState(group, node, battle) {
      const key = turnKey(battle.agent_id, battle.battle_id, group.requestSequence);
      const state = tracePaneState.get(key);
      if (!state || state.atBottom || group.latest.turn_state === "thinking") {
        node.scrollTop = node.scrollHeight;
        return;
      }
      node.scrollTop = state.scrollTop;
    }

    function renderHeroActions(battle) {
      const actions = document.getElementById("hero-actions");
      actions.innerHTML = "";
      if (!battle) return;
      if (battle.finished) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ghost-button danger-button";
        button.textContent = "Delete session";
        button.addEventListener("click", async () => {
          const confirmed = window.confirm(`Move session ${battle.battle_id} to Trash?`);
          if (!confirmed) return;
          try {
            await deleteBattle(battle.agent_id, battle.battle_id);
            selectedAgent = null;
            selectedBattle = null;
            window.location.hash = "";
            await refresh();
          } catch (error) {
            window.alert(error instanceof Error ? error.message : String(error));
          }
        });
        actions.appendChild(button);
        return;
      }
      if (battle.transport !== "showdown-client") return;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "ghost-button danger-button";
      button.textContent = battle.stop_requested_at ? "Stopping…" : "Stop agent";
      button.disabled = Boolean(battle.stop_requested_at);
      if (!battle.stop_requested_at) {
        button.addEventListener("click", async () => {
          const confirmed = window.confirm(`Forfeit ${battle.battle_id} and stop the agent from making any more decisions in this battle?`);
          if (!confirmed) return;
          try {
            await stopBattle(battle.agent_id, battle.battle_id);
            await refresh();
          } catch (error) {
            window.alert(error instanceof Error ? error.message : String(error));
          }
        });
      }
      actions.appendChild(button);
    }

    function messageGroupKey(group) {
      return `${group.entryKind}:${group.requestSequence}`;
    }

    function buildSummaryCardMarkup(latest) {
      const helperSummary = helperSummaryText(latest.summary.helper_summary || latest.summary.helperSummary);
      const costEstimate = latest.summary.cost_estimate || latest.summary.costEstimate || null;
      const costMeta = [];
      if (costEstimate && typeof costEstimate.model === "string" && costEstimate.model) {
        costMeta.push(costEstimate.model);
      }
      if (costEstimate && typeof costEstimate.pricing_source === "string" && costEstimate.pricing_source) {
        costMeta.push(costEstimate.pricing_source);
      }
      return `
        <div class="turn-header">
          <div>
            <h3 class="turn-title">Battle summary</h3>
            <div class="turn-meta">${escapeHtml(formatAbsoluteTimestamp(latest.created_at))}</div>
          </div>
          <div class="header-actions">
            <span class="chip state-chip state-accepted">completed</span>
          </div>
        </div>
        <div class="summary-grid">
          <div class="summary-item"><strong>Result</strong><div>${escapeHtml(String(latest.summary.result || "unknown"))}</div></div>
          <div class="summary-item"><strong>Total turns</strong><div>${escapeHtml(String(latest.summary.total_turns ?? "n/a"))}</div></div>
          <div class="summary-item"><strong>Average decision time</strong><div>${escapeHtml(formatDurationMs(latest.summary.average_decision_latency_ms))}</div></div>
          <div class="summary-item"><strong>Tokens</strong><div>${escapeHtml(formatUsage(latest.summary.usage || latest.usage))}</div></div>
          <div class="summary-item"><strong>Estimated cost</strong><div>${escapeHtml(formatUsdCost(costEstimate))}</div>${costMeta.length ? `<div class="turn-meta">${escapeHtml(costMeta.join(" · "))}</div>` : ""}</div>
        </div>
        ${helperSummary ? `<div class="summary">Helpers used · ${escapeHtml(helperSummary)}</div>` : ""}
      `;
    }

    function buildTurnCardMarkup(group, latest, battle, traceExpanded, key) {
      const stateClass = String(latest.turn_state || "thinking").replace(/_/g, "-");
      const helperEvents = summarizeHelpers(group);
      const traceTextValue = traceText(group);
      const summaryBits = [];
      if (latest.submission_detail) summaryBits.push(latest.submission_detail);
      if (latest.error) summaryBits.push(`error: ${latest.error}`);
      if (latest.notes) summaryBits.push(`notes: ${latest.notes}`);
      if (latest.decision_latency_ms !== null) summaryBits.push(`latency: ${formatDurationMs(latest.decision_latency_ms)}`);
      if (latest.fallback_reason === "timeout" && latest.timeout_seconds) {
        summaryBits.push(`timed out after ${latest.timeout_seconds}s`);
      } else if (latest.fallback_used && latest.fallback_reason) {
        summaryBits.push(`fallback: ${latest.fallback_reason}`);
      }
      if (latest.usage && typeof latest.usage.total_tokens === "number") {
        summaryBits.push(`tokens: ${latest.usage.total_tokens.toLocaleString()}`);
      }

      return `
        <div class="turn-header">
          <div>
            <h3 class="turn-title">Turn ${latest.turn_number ?? "setup"} · Request ${latest.request_sequence}</h3>
            <div class="turn-meta">Latest attempt ${latest.decision_attempt} · ${escapeHtml(latest.request_kind)} · rqid ${escapeHtml(latest.rqid || "none")} · ${escapeHtml(formatAbsoluteTimestamp(latest.created_at))}</div>
          </div>
          <div class="header-actions">
            <span class="chip state-chip state-${escapeHtml(stateClass)}">${escapeHtml(displayTurnState(latest))}</span>
            <button class="ghost-button prompt-button" type="button">Prompt</button>
          </div>
        </div>
        <div class="chip-row">
          <span class="chip">${escapeHtml(latest.request_kind)}</span>
          <span class="chip">attempt ${escapeHtml(String(latest.decision_attempt))}</span>
          <span class="chip">${escapeHtml(latest.submission_state || "pending")}</span>
          ${latest.selected_action_source ? `<span class="chip">${escapeHtml(latest.selected_action_source)}</span>` : ""}
        </div>
        ${helperEvents.length ? `<div class="helper-strip">${helperEvents.map(helper => `<span class="chip">helper ${escapeHtml(helper.name)} ×${escapeHtml(String(helper.count))}</span>`).join("")}</div>` : ""}
        <div class="summary">${escapeHtml(summaryBits.join(" · ") || (latest.turn_state === "thinking" ? "Agent is still thinking." : "Turn recorded."))}</div>
        <div class="decision-box">
          <strong>Action taken</strong>
          <div class="decision-line">${escapeHtml(latest.selected_action_label || latest.selected_action || "No action selected yet")}</div>
          <div class="turn-meta">${escapeHtml(latest.selected_action_source || "pending selection")}</div>
        </div>
        <div class="trace-controls">
          <strong>Live trace</strong>
          <button class="ghost-button trace-toggle" type="button">${traceExpanded ? "Hide trace" : "Show trace"}</button>
        </div>
        <div class="trace-shell ${traceExpanded ? "" : "collapsed"}">
          <pre class="trace-log" data-trace-key="${escapeHtml(key)}">${escapeHtml(traceTextValue)}</pre>
        </div>
      `;
    }

    function upsertMessageCard(messages, battle, group) {
      const latest = group.latest;
      const key = messageGroupKey(group);
      let card = messages.querySelector(`[data-message-key="${CSS.escape(key)}"]`);
      if (!card) {
        card = document.createElement("section");
        card.dataset.messageKey = key;
      }

      card.className = "turn-card";
      if (latest.turn_state === "thinking") {
        card.classList.add("live");
      }

      if (group.entryKind === "summary" && latest.summary) {
        card.innerHTML = buildSummaryCardMarkup(latest);
        return card;
      }

      const traceExpanded = latest.turn_state === "thinking" || expandedTraces.has(key);
      card.innerHTML = buildTurnCardMarkup(group, latest, battle, traceExpanded, turnKey(battle.agent_id, battle.battle_id, group.requestSequence));

      card.querySelector(".prompt-button").addEventListener("click", () => {
        openPromptModal(latest);
      });
      card.querySelector(".trace-toggle").addEventListener("click", () => {
        if (expandedTraces.has(key)) {
          expandedTraces.delete(key);
        } else {
          expandedTraces.add(key);
        }
        refresh();
      });

      const log = card.querySelector(".trace-log");
      if (log) {
        restoreTracePaneState(group, log, battle);
      }
      return card;
    }

    function reconcileMessageCards(messages, battle, groups) {
      const desiredKeys = new Set();
      for (const group of groups) {
        const key = messageGroupKey(group);
        desiredKeys.add(key);
        const card = upsertMessageCard(messages, battle, group);
        messages.appendChild(card);
      }

      Array.from(messages.children).forEach(node => {
        if (!(node instanceof HTMLElement)) return;
        const key = node.dataset.messageKey;
        if (!key) {
          node.remove();
          return;
        }
        if (!desiredKeys.has(key)) {
          node.remove();
        }
      });
    }

    function renderMessages(battle) {
      const title = document.getElementById("battle-title");
      const subtitle = document.getElementById("battle-subtitle");
      const messages = document.getElementById("messages");
      renderHeroActions(battle);
      if (!battle) {
        title.textContent = "No session selected";
        subtitle.textContent = "Pick a session from the left once a transcript exists.";
        messages.innerHTML = '<div class="empty">Session details will appear here once the selected battle has recorded activity.</div>';
        return;
      }

      title.textContent = `${battle.battle_id} · ${battle.format_name || "Unknown format"}`;
      subtitle.textContent = `${battle.agent_id} vs ${battle.challenger || "local player"} · ${statusLabel(battle)}${battle.finished && battle.winner ? ` · winner ${battle.winner}` : ""}`;

      if (!Array.isArray(battle.entries) || !battle.entries.length) {
        messages.innerHTML = '<div class="empty">This battle has started, but the agent has not reached a recorded decision turn yet.</div>';
        return;
      }

      const groups = groupEntries(battle.entries);
      reconcileMessageCards(messages, battle, groups);
    }

    function openPromptModal(entry) {
      document.getElementById("prompt-title").textContent = `Prompt · Turn ${entry.turn_number ?? "setup"} · Request ${entry.request_sequence}`;
      const timestampBits = [
        `Created ${formatAbsoluteTimestamp(entry.created_at)}`,
      ];
      if (entry.submitted_at) timestampBits.push(`Submitted ${formatAbsoluteTimestamp(entry.submitted_at)}`);
      if (entry.timed_out_at) timestampBits.push(`Timed out ${formatAbsoluteTimestamp(entry.timed_out_at)}`);
      if (entry.validated_at) timestampBits.push(`Validated ${formatAbsoluteTimestamp(entry.validated_at)}`);
      document.getElementById("prompt-meta").textContent = `${entry.request_kind} · attempt ${entry.decision_attempt} · rqid ${entry.rqid || "none"} · ${timestampBits.join(" · ")}`;
      document.getElementById("prompt-body").textContent = entry.prompt_text || "";
      document.getElementById("prompt-history").textContent = (entry.recent_public_events || []).join("\\n");
      document.getElementById("prompt-modal").classList.remove("hidden");
    }

    function closePromptModal() {
      document.getElementById("prompt-modal").classList.add("hidden");
    }

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    async function refresh() {
      readSelectionFromHash();
      const main = document.getElementById("main");
      const sidebar = document.getElementById("sidebar");
      const windowScrollX = window.scrollX;
      const windowScrollY = window.scrollY;
      const mainScrollTop = main.scrollTop;
      const sidebarScrollTop = sidebar.scrollTop;
      captureTracePaneState();
      const payload = await fetchBattles();
      const battles = payload.battles || [];
      renderBattleList(battles);
      if (!selectedAgent || !selectedBattle) {
        renderMessages(null);
        main.scrollTop = mainScrollTop;
        sidebar.scrollTop = sidebarScrollTop;
        return;
      }
      const battle = await fetchBattle(selectedAgent, selectedBattle);
      if (battle === null) {
        const fallback = battles[0] || null;
        if (fallback) {
          selectedAgent = fallback.agent_id;
          selectedBattle = fallback.battle_id;
          updateHash(selectedAgent, selectedBattle);
          renderMessages(await fetchBattle(selectedAgent, selectedBattle));
        } else {
          renderMessages(null);
        }
      } else {
        renderMessages(battle);
      }
      window.scrollTo(windowScrollX, windowScrollY);
      main.scrollTop = mainScrollTop;
      sidebar.scrollTop = sidebarScrollTop;
    }

    document.getElementById("prompt-close").addEventListener("click", closePromptModal);
    document.getElementById("prompt-modal").addEventListener("click", (event) => {
      if (event.target.id === "prompt-modal") {
        closePromptModal();
      }
    });

    window.addEventListener("hashchange", refresh);
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""
