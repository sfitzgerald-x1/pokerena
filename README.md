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

`config/agents.yaml` is reserved for future agent definitions. The first scaffold validates and surfaces these entries without launching them yet.

## Notes

- Pokerena targets Python 3.14 for future multithreaded orchestration work.
- Pokemon Showdown itself still requires Node.js 22+ and npm.
- Generated runtime files live under `.runtime/showdown/`.
