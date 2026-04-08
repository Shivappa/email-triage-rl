"""
server/app.py — FastAPI application entry point for the Email Triage environment.

Supports TASK_LEVEL environment variable: easy | medium | hard  (default: easy)
"""

import os
from openenv.core.env_server.http_server import create_app

try:
    from ..models import TriageAction, TriageObservation
    from .my_environment import EmailTriageEnvironment
except ModuleNotFoundError:
    from email_triage_rl_hackathon.models import TriageAction, TriageObservation
    from email_triage_rl_hackathon.server.my_environment import EmailTriageEnvironment

task_level = os.environ.get("TASK_LEVEL", "easy")

app = create_app(
    EmailTriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="email_triage",
    max_concurrent_envs=10,
)


def main(host: str = "0.0.0.0", port: int = int(os.environ.get("PORT", 8000))):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()  # picks up PORT env var; override with --host / --port if needed
