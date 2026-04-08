---
title: Email Triage RL Environment
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - email-triage
  - rl
  - real-world
---

# 📧 Email Triage Environment

> **Meta × PyTorch OpenEnv Hackathon — Round 1**
> An RL environment where an AI agent learns to triage real-world emails.

## 🎯 What Is This?

This environment simulates an **email triage system** — a task every business performs daily. An agent receives a queue of emails and must:

1. **Classify** the email (spam / billing / technical / hr / general)
2. **Assign a priority** (low / medium / high / urgent)
3. **Route** to the correct department (support / finance / engineering / hr / management)
4. **Draft a professional reply** (required for medium and hard tasks)

Rewards are **continuous and partial** — agents earn credit for getting each dimension right, incentivising learning even when the full answer isn't perfect.

---

## 🗂️ Action Space

| Field        | Type   | Values                                              |
|--------------|--------|-----------------------------------------------------|
| `category`   | string | `spam` · `billing` · `technical` · `hr` · `general` |
| `priority`   | string | `low` · `medium` · `high` · `urgent`                |
| `department` | string | `support` · `finance` · `engineering` · `hr` · `management` |
| `reply`      | string | Free-text reply draft (empty string for easy tasks) |

---

## 👁️ Observation Space

| Field             | Type   | Description                                  |
|-------------------|--------|----------------------------------------------|
| `email_id`        | string | Unique email identifier                      |
| `subject`         | string | Email subject line                           |
| `body`            | string | Full email body                              |
| `sender`          | string | Sender's email address                       |
| `task_level`      | string | `easy` · `medium` · `hard`                  |
| `feedback`        | string | Natural-language grader feedback             |
| `score`           | float  | Cumulative average score (0.0–1.0)           |
| `emails_remaining`| int    | Number of emails left in the episode         |
| `done`            | bool   | Episode complete flag                        |
| `reward`          | float  | Per-step reward (0.0–1.0)                    |

---

## 📊 Task Descriptions

### 🟢 Easy — Classification Only
- **5 emails** per episode (spam, billing, technical, HR, general)
- Agent must correctly classify category, priority, and department
- No reply required
- Score: `category(0.4) + priority(0.3) + department(0.3)`

### 🟡 Medium — Classification + Short Reply
- **3 emails** per episode (billing urgency, technical incident, HR complaint)
- Agent must classify AND write a professional 2-4 sentence reply
- Score: `classification(0.5) + reply_quality(0.5)`
- Reply graded on: professional tone, keyword coverage

### 🔴 Hard — Classification + Detailed Accurate Reply
- **3 emails** per episode (enterprise escalations, security incidents, board complaints)
- Agent must classify AND write a detailed, accurate reply covering specific required points
- Score: `classification(0.4) + reply_depth(0.6)`
- Reply graded on: length, tone, keyword coverage (9 required keywords per email)

---

## 🏆 Reward Function

Rewards are **partial and continuous** — agents earn credit at every step:

| Condition                                 | Reward        |
|-------------------------------------------|---------------|
| Correct category                          | +0.15 – +0.40 |
| Correct priority                          | +0.13 – +0.30 |
| Correct department                        | +0.12 – +0.30 |
| Reply provided (non-trivial)              | +0.05 – +0.10 |
| Professional tone in reply                | +0.05 – +0.10 |
| Each required keyword matched             | proportional  |
| Missing reply on medium/hard             | −0.10 penalty |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
git clone https://huggingface.co/spaces/Shivacode/rl-hackathon
cd email-triage-env
pip install -e .
```

### 2. Run the server locally
```bash
# Easy task (default)
uvicorn email_triage_rl_hackathon.server.app:app --host 0.0.0.0 --port 8000 --reload

# Medium task
TASK_LEVEL=medium uvicorn email_triage_rl_hackathon.server.app:app --port 8000

# Hard task
TASK_LEVEL=hard uvicorn email_triage_rl_hackathon.server.app:app --port 8000
```

### 3. Interact with the environment (Python)
```python
import requests

# Reset
obs = requests.post("http://localhost:8000/reset", json={"seed": 42}).json()
print(obs["observation"]["subject"])

# Step
action = {
    "action": {
        "category": "billing",
        "priority": "urgent",
        "department": "finance",
        "reply": "Dear customer, we apologize for the inconvenience and will investigate within 24 hours."
    }
}
result = requests.post("http://localhost:8000/step", json=action).json()
print(result["reward"])
print(result["observation"]["feedback"])
```

### 4. Run with Docker
```bash
cd email_triage_rl_hackathon
docker build -t email-triage-env .
docker run -p 7860:7860 -e TASK_LEVEL=easy email-triage-env
```

### 5. Run baseline inference
```bash
export OPENAI_API_KEY=your-key-here
python baseline.py
# Scores saved to outputs/evals/baseline_results.json
```

---

## 📈 Baseline Scores

Measured with `gpt-4o-mini` at temperature 0:

| Task Level | Score  |
|------------|--------|
| Easy       | ~0.90  |
| Medium     | ~0.72  |
| Hard       | ~0.55  |
| **Overall**| **~0.72** |

---

## 🗃️ Project Structure

```
email_triage_rl_hackathon/
├── __init__.py                   # Public exports
├── models.py                     # TriageAction, TriageObservation
├── client.py                     # EmailTriageEnv (EnvClient)
├── openenv.yaml                  # Environment manifest
├── pyproject.toml                # Package config
├── Dockerfile                    # HF Spaces / Docker deployment
├── README.md                     # This file
├── outputs/
│   ├── logs/
│   └── evals/baseline_results.json
└── server/
    ├── my_environment.py                # EmailTriageEnvironment (reset/step/state)
    ├── email_data.py             # 11 curated emails across 3 levels
    ├── graders.py                # Deterministic graders (easy/medium/hard)
    ├── app.py                    # FastAPI app
    └── requirements.txt
```

---

## 🔑 Environment Variables

| Variable        | Default                    | Description                          |
|-----------------|----------------------------|--------------------------------------|
| `TASK_LEVEL`    | `easy`                     | Task difficulty: easy/medium/hard    |
| `PORT`          | `7860`                     | Server port                          |
| `OPENAI_API_KEY`| —                          | Required for baseline script         |
| `MODEL_NAME`    | `gpt-4o-mini`              | Model for baseline                   |
| `ENV_BASE_URL`  | `http://localhost:8000`    | Environment URL for baseline         |

