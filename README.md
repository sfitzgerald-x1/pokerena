# Pokérena

Pokérena is a Python-led harness for running a local Pokemon Showdown server that can later host LLM-powered battle agents. This first scaffold focuses on standing up a pinned local server with a clean config layer for future agent integrations.

## What This PR Sets Up

- A pinned Pokemon Showdown submodule under `vendor/pokemon-showdown`
- A Python 3.14 CLI for environment checks, generated config rendering, and server startup
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
   ./scripts/bootstrap-showdown.sh
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

Start the local server:

```bash
python3.14 -m pokerena server up --config config/server.local.yaml
```

By default, the local server will be reachable at http://localhost:8000.

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
- `hook.type`
- `hook.context_format`
- `hook.decision_format`
- `hook.prompt_style`

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

The live runtime applies a per-decision timeout and a max invalid-choice retry policy. If the agent times out or keeps returning illegal choices, Pokérena falls back to the built-in `first-legal` policy for that request so the battle can continue deterministically.

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

### Adapter Status

- `sim-stream` is the first end-to-end adapter and is the default local development path.
- `showdown-client` is defined as a future adapter shape, but it is not implemented yet.
- Public main-server play is intentionally not enabled by default in this scaffold.

## Notes

- Pokerena targets Python 3.14 for future multithreaded orchestration work.
- Pokemon Showdown itself still requires Node.js 22+ and npm.
- Generated runtime files live under `.runtime/showdown/`.
