# Pokérena

Pokérena is a Python-led harness for running a local Pokemon Showdown server and wiring local battle agents into it. The current scaffold stands up a pinned local server, exposes a stable subprocess hook for agents, and can now bring callable local bots online so you can challenge them from the browser.

## What This PR Sets Up

- A pinned Pokemon Showdown submodule under `vendor/pokemon-showdown`
- A repo-owned damage calc wrapper built on the official `@smogon/calc` package
- A Python 3.14 CLI for environment checks, generated config rendering, server startup, and agent runtimes
- YAML config files for local server settings and future agent definitions
- Host-run and Docker startup paths that use the same Pokerena config

## Prerequisites

- Python 3.14
- Node.js 22+
- npm
- Docker Desktop or Docker Engine if you want the container path

## Initial Setup

1. Initialize the upstream Showdown checkout and install its dependencies:

   ```bash
   ./scripts/bootstrap-node-deps.sh
   ```

2. Copy the example configs into local working files:

   ```bash
   cp config/server.local.example.yaml config/server.local.yaml
   cp config/agents.example.yaml config/agents.yaml
   cp .env.example .env
   ```

3. Create a Python 3.14 virtual environment and install Pokerena:

   ```bash
   python3.14 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install .
   ```

4. Verify the local environment:

   ```bash
   python3.14 -m pokerena server doctor --config config/server.local.yaml --agents-config config/agents.yaml
   ```

## Host Run Flow

Render the generated Showdown config into `.runtime/showdown/`:

```bash
python3.14 -m pokerena server render-config --config config/server.local.yaml
```

Preview the command without starting the process:

```bash
python3.14 -m pokerena server up --config config/server.local.yaml --dry-run
```

Start the local server and any enabled callable local agents:

```bash
python3.14 -m pokerena server up --config config/server.local.yaml --agents-config config/agents.yaml
```

By default, the local server will be reachable at http://localhost:8000. If an agent has `callable.enabled: true`, `server up` will also start that bot process, wait for the server to come online, and keep the child processes tied to the server lifecycle.

## Docker Flow

After copying `config/server.local.yaml`, `config/agents.yaml`, and `.env`, build and run the containerized server:

```bash
docker compose up --build
```

This mounts your local config files read-only and writes generated runtime state into `.runtime/`.

## Config Files

`config/server.local.yaml` controls the local server scaffold:

- `showdown_path`
- `bind_address`
- `port`
- `server_id`
- `public_origin`
- `no_security`
- `data_dir`
- `log_dir`

`config/agents.yaml` now defines the first local battle-agent runtime. Each entry declares:

- `provider`
- `player_slot`
- `transport`
- `launch.command`
- `launch.args`
- `env_file`
- `hook.type`
- `hook.context_format`
- `hook.decision_format`
- `hook.prompt_style`
- `callable.enabled`
- `callable.username`
- `callable.accepted_formats`
- `callable.challenge_policy`
- `callable.avatar`

Keep checked-in configs free of secrets. Use `.env` or an agent-specific `env_file` for anything sensitive or machine-local. The local callable flow in this repo does not require a password or registered Showdown account; it relies on `no_security: true` plus a local rename.

## Damage Calc Wrapper

Pokérena exposes damage calculation through its own CLI instead of asking agents to call Node directly. The wrapper delegates to the official `@smogon/calc` package locally, keeps a small worker alive under `.runtime/calc/`, and returns stable JSON.

Use a JSON file:

```bash
python3.14 -m pokerena calc damage --input damage-request.json
```

Or pipe the request on stdin:

```bash
cat damage-request.json | python3.14 -m pokerena calc damage --stdin
```

For move comparisons, batch multiple requests in one call:

```bash
cat damage-batch-request.json | python3.14 -m pokerena calc damage-batch --stdin
```

The input shape is intentionally narrow for v1:

```json
{
  "schema_version": "pokerena.damage-request.v1",
  "generation": 2,
  "attacker": {
    "species": "Snorlax",
    "options": {
      "level": 100,
      "item": "Leftovers"
    }
  },
  "defender": {
    "species": "Raikou",
    "options": {
      "level": 100,
      "item": "Leftovers"
    }
  },
  "move": {
    "name": "Double-Edge"
  },
  "field": {}
}
```

The wrapper returns JSON with:

- `schema_version`
- `damage`
- `range`
- `range_percent`
- `description`
- `knockout`

Agents should call `python3.14 -m pokerena calc damage` or `python3.14 -m pokerena calc damage-batch`, not `node`, so the command surface stays stable even if the underlying Node implementation changes later.

The deprecated `./scripts/bootstrap-showdown.sh` path still forwards to `./scripts/bootstrap-node-deps.sh` for now, so existing local setup notes keep working while the new name settles in.

## Agent Runtime

Pokérena now uses an event-driven battle session core instead of parsing public battle logs. The first live adapter is the local Showdown simulator stream, which emits:

- public battle updates
- private per-player `|request|` payloads
- battle end metadata

That means the agent sees the same decision surface that Showdown itself uses. Pokérena owns the session state, retry counters, and recent public history, while legality is forwarded directly from the raw `request` payload instead of being recomputed.

The stable turn context schema is `pokerena.turn-context.v1`. It includes:

- `rqid` as the request identity when available, with a stable synthetic ID in local simulator mode
- `signals.turn_started` as informational metadata
- `signals.request_updated` and `signals.decision_required` as the operational decision signals
- the raw Showdown `request` payload
- recent public protocol lines
- simple legal action hints derived from the request payload

The decision schema is `pokerena.decision.v1`, which returns one Showdown choice string plus optional notes.

### Live Simulator Flow

Run a local simulator battle with the configured agent on one side and a built-in `first-legal` fallback opponent on the other:

```bash
python3.14 -m pokerena agent sim-battle \
  --config config/server.local.yaml \
  --agents-config config/agents.yaml \
  --agent-id example-randbat-bot \
  --format gen9randombattle
```

Use `--dry-run` to stop after the first turn context is rendered:

```bash
python3.14 -m pokerena agent sim-battle \
  --config config/server.local.yaml \
  --agents-config config/agents.yaml \
  --agent-id example-randbat-bot \
  --format gen9randombattle \
  --dry-run
```

The live runtime applies a per-decision timeout and a max invalid-choice retry policy. Timeouts fall back to a random legal action for the current request shape, while repeated invalid choices still fall back to the built-in `first-legal` policy so the battle can continue.

### Local Browser Challenges

For human-vs-agent testing on your local Showdown server, set an agent to `transport: showdown-client` and enable its `callable` block. The bot logs in locally with `/trn USERNAME` under `--no-security`, so you do not need to store a Showdown password in the repo.

Example callable block:

```yaml
callable:
  enabled: true
  username: LocalAgentBot
  accepted_formats:
    - gen3randombattle
  challenge_policy: accept-direct-challenges
```

You can also tune how much recent battle history the hook receives on each turn, and how long Pokerena waits before timing out the turn:

```yaml
hook:
  history_turn_limit: 4
  decision_timeout_seconds: 120
```

Then start the server:

```bash
python3.14 -m pokerena server up --config config/server.local.yaml --agents-config config/agents.yaml
```

Open the local Showdown UI in your browser, challenge your configured bot username to `gen3randombattle`, and the bot will auto-accept matching direct challenges. Unsupported formats are rejected with a PM, and bot names stay local to your machine unless you later add env-backed auth for another environment.

When the transcript viewer is enabled, `server up` also starts a separate local Battle Sessions page on `http://localhost:8001` by default. Open it beside the battle to watch newest-first turn cards, a live fixed-height trace while the agent is thinking, compact helper activity, and a prompt modal for the full context Pokerena passed into the hook. Live showdown-client sessions can be stopped from the page, which cancels the current agent turn and forfeits the battle, and finished sessions can be moved to Trash directly from the same UI. Finished sessions also get a summary card with result, total turns, average decision time, and any token totals the provider exposed.

If you want to run the bot transport directly without `server up`, use:

```bash
python3.14 -m pokerena agent showdown-client \
  --config config/server.local.yaml \
  --agents-config config/agents.yaml \
  --agent-id example-randbat-bot
```

### Replay And Debugging

Pokérena writes normalized battle captures under `.runtime/agents/<agent-id>/<battle-id>/capture.json`. These captures contain both public updates and private side requests, so they can be replayed later without relying on incomplete public transcripts.

To inspect the current turn context from a recorded capture:

```bash
python3.14 -m pokerena agent context \
  --agent-id example-randbat-bot \
  --agents-config config/agents.yaml \
  --capture .runtime/agents/example-randbat-bot/<battle-id>/capture.json
```

To rebuild the same context and invoke the configured subprocess hook from a capture:

```bash
python3.14 -m pokerena agent decide \
  --agent-id example-randbat-bot \
  --agents-config config/agents.yaml \
  --capture .runtime/agents/example-randbat-bot/<battle-id>/capture.json
```

The hook writes its exact turn context, prompt, response, and cursor state into `.runtime/agents/<agent-id>/<battle-id>/`.

Pokérena also writes `.runtime/agents/<agent-id>/<battle-id>/transcript.json`, which is the canonical structured source for the local Battle Sessions viewer. Battle session storage stays file-based under `.runtime/agents/<agent-id>/<battle-id>/`, with `capture.json`, `cursor.json`, `transcript.json`, and the latest turn artifacts side by side. Deleting a finished session from the UI moves that directory to the OS Trash instead of deleting it permanently.

### Adapter Status

- `sim-stream` is the first end-to-end adapter and is the default local development path.
- `showdown-client` now supports local direct challenges on the local Pokerena Showdown server.
- Public main-server play is intentionally not enabled by default in this scaffold.

## Notes

- Pokerena targets Python 3.14 for future multithreaded orchestration work.
- Pokemon Showdown and the local calc wrapper both require Node.js 22+ and npm.
- Generated runtime files live under `.runtime/showdown/`.
