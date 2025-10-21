Ollama + Collector Docker Compose (located in `test2/ollama-setup`)

This directory contains a Docker Compose setup to run Ollama (local LLM server) and an optional collector container that runs your data-collection scripts against Ollama.

Files:
- `docker-compose.yml`: orchestrates `ollama` and `collector` services.
- `Dockerfile`: used to build the `collector` image. It installs dependencies from the repo `requirements.txt`.

Quick start (from repo root):

1) Change into the `test2/ollama-setup` directory and start the services (detached):

```bash
cd test2/ollama-setup
docker compose up -d
```

2) Watch Ollama logs while model downloads:

```bash
docker compose logs -f ollama
```

3) Verify the API from the host (after model pull finishes):

```bash
curl http://localhost:11434/api/tags
```

4) Run the collector to execute your `clip-rl-2.py` collection CLI (example: show help):

```bash
docker compose run --rm collector
```

To actually run collection replace the collector entrypoint in `docker-compose.yml` or run:

```bash
docker compose run --rm collector bash -lc "python3 test2/clip-rl-2.py collect --episodes 100 --output_dir /workspace/test2/collected"
```

Notes:
- The collector mounts the repository root at `/workspace` so code changes on host are visible immediately inside the container.
- Ollama stores model files in the named volume `ollama_data` so models persist across restarts.
- For GPU acceleration, install NVIDIA Container Toolkit and enable GPU support in the compose file or run the container with `--gpus all`.

Troubleshooting:
- If the model does not show up in `/api/tags`, check `docker compose logs -f ollama` and ensure the pull completed.
- If you get port conflicts, change `11434` in `docker-compose.yml`.

