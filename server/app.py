"""
server/app.py — Root-level shim for openenv validate compatibility.

The real application lives in email_triage_rl_hackathon/server/app.py.
This file exists so that `openenv validate` (which checks for server/app.py
at the repo root) passes when the validator clones the GitHub repo.
"""

import os
from email_triage_rl_hackathon.server.app import app  # noqa: F401


def main(host: str = "0.0.0.0", port: int = int(os.environ.get("PORT", 7860))):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
