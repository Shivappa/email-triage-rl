# Hackathon — Meta x PyTorch OpenEnv

## Structure

```
hackathon-sst/
├── email_triage_rl_hackathon/              
│   ├── models.py        ← Action / Observation / State
│   ├── client.py        ← EnvClient
│   ├── server/
│   │   ├── my_environment.py          ← reset(), step(), state()
│   │   └── app.py             ← FastAPI server
│   ├── Dockerfile
│   └── openenv.yaml
├── pyproject.toml
└── .venv/               ← Python virtual environment
```

## Getting Started

```bash
# Activate venv
source .venv/bin/activate

# Install dependencies
pip install openenv-core fastapi uvicorn pydantic

# Run the server
uvicorn email_triage_rl_hackathon.server.app:app --reload --port 8000
```

## Useful Links

- [OpenEnv GitHub](https://github.com/meta-pytorch/OpenEnv)
- [OpenEnv Course](https://github.com/huggingface/openenv-course)
- [OpenEnv Docs](https://meta-pytorch.org/OpenEnv/)
- [Hackathon Dashboard](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/)
