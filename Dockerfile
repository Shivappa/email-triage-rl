# ──────────────────────────────────────────────────────────────
# Official Meta × PyTorch OpenEnv base image
#   • Python 3.11.15 pre-installed
#   • FastAPI + uvicorn pre-installed
#   • WORKDIR /app, PYTHONUNBUFFERED=1, UV_SYSTEM_PYTHON=1 set
#   • Uses uv as the fast package manager
# ──────────────────────────────────────────────────────────────
FROM ghcr.io/meta-pytorch/openenv-base:latest

# ── System dependencies (curl for HEALTHCHECK) ────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python path: include both /app and /app/src ───────────────
# Base image sets PYTHONPATH=/app/src; we extend it so the
# top-level package (email_triage_rl_hackathon/) is importable
# whether running from /app or /app/src.
ENV PYTHONPATH=/app:/app/src

# ── Enable the built-in OpenEnv web inspector UI ──────────────
# Set to "false" to disable the Swagger/ReDoc interface.
ENV ENABLE_WEB_INTERFACE=true

# ── Runtime config ────────────────────────────────────────────
ENV TASK_LEVEL=easy
ENV PORT=7860

# ── Install only what the base image does NOT already provide ─
# openenv-core is not pre-installed in the base image.
# fastapi + uvicorn are already present — uv skips them.
COPY email_triage_rl_hackathon/server/requirements.txt ./requirements.txt
RUN uv pip install --no-cache -r requirements.txt

# ── Copy package source and install it ────────────────────────
COPY email_triage_rl_hackathon/ ./email_triage_rl_hackathon/
RUN uv pip install --no-cache ./email_triage_rl_hackathon/

# ── Runtime output directories ────────────────────────────────
RUN mkdir -p email_triage_rl_hackathon/outputs/logs \
             email_triage_rl_hackathon/outputs/evals

EXPOSE 7860

# ── Health check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn email_triage_rl_hackathon.server.app:app --host 0.0.0.0 --port ${PORT}"]
