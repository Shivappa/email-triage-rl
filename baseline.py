"""
baseline.py — Baseline inference script for the Email Triage environment.

Runs an OpenAI-compatible model against all three task levels via the WebSocket client
and reports reproducible scores.

Reads credentials from:
  API_KEY         (required — injected by hackathon validator)
  API_BASE_URL    (required — injected by hackathon validator)
  MODEL_NAME      (optional, default: Qwen/Qwen2.5-72B-Instruct)
  ENV_BASE_URL    (optional, default: http://localhost:8000)

Usage:
    # 1. Start the server:
    uvicorn email_triage_rl_hackathon.server.app:app --port 8000

    # 2. Run the baseline:
    API_KEY=<key> API_BASE_URL=<url> python baseline.py

    # Or against a deployed HF Space:
    ENV_BASE_URL=https://Shivacode-rl-hackathon.hf.space API_KEY=<key> API_BASE_URL=<url> python baseline.py
"""

import os
import json
import sys
import time

# Auto-load .env file if present (pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — fall back to real env vars

from openai import OpenAI
from email_triage_rl_hackathon.client import EmailTriageEnv
from email_triage_rl_hackathon.models import TriageAction

# ── Config ────────────────────────────────────────────────────────
# API_KEY and API_BASE_URL are injected by the hackathon validator — always use them first.
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
BASE_URL     = os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")   # used by openenv push / huggingface_hub

if not API_KEY:
    print("ERROR: API_KEY not set. The hackathon validator injects API_KEY automatically.")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── System prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert email triage agent.
For each email you receive, respond with a JSON object with EXACTLY these fields:
{
  "category":   one of ["spam", "billing", "technical", "hr", "general"],
  "priority":   one of ["low", "medium", "high", "urgent"],
  "department": one of ["support", "finance", "engineering", "hr", "management"],
  "reply":      "your professional reply draft (empty string for easy tasks)"
}

Rules:
- spam emails: low priority, route to support, no reply needed
- billing issues (payments, invoices, refunds): finance department
- technical issues (bugs, login, API, outages): engineering department
- HR matters (leave, harassment, complaints): hr department
- board/executive escalations: management department
- For medium/hard tasks, write a professional reply (2-4 sentences minimum)
- Urgent issues need immediate acknowledgment in the reply
Respond ONLY with the JSON object, no other text."""


def build_user_prompt(obs) -> str:
    return f"""Task level: {obs.task_level}
Email ID: {obs.email_id}
From: {obs.sender}
Subject: {obs.subject}

{obs.body}

Emails remaining after this one: {obs.emails_remaining}
{"Note: Write a professional reply for this task level." if obs.task_level in ("medium", "hard") else "Note: Classification only (reply can be empty)."}"""


def call_model(obs) -> TriageAction:
    """Call the OpenAI model and parse the JSON action."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(obs)},
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content)
        return TriageAction(**data)
    except Exception as e:
        print(f"  [model error] {e} — using fallback action")
        return TriageAction(category="general", priority="low",
                            department="support", reply="")


def run_task(task_level: str) -> float:
    """Run one full episode for the given task level using the WS client. Returns average score."""
    print(f"\n{'='*60}")
    print(f"  TASK LEVEL: {task_level.upper()}")
    print(f"{'='*60}")

    total_reward = 0.0
    steps = 0

    with EmailTriageEnv(base_url=ENV_BASE_URL).sync() as env:
        obs = env.reset(seed=42, task_level=task_level)

        while not obs.done and obs.email_id != "none":
            print(f"\n  📧 [{obs.email_id}] {obs.subject[:60]}")

            action = call_model(obs)
            print(f"     → category={action.category}, priority={action.priority}, "
                  f"department={action.department}")
            if action.reply:
                preview = action.reply[:80].replace("\n", " ")
                print(f"     → reply: {preview}...")

            result = env.step(action)
            reward = result.reward or 0.0
            total_reward += reward
            steps += 1

            obs = result.observation
            feedback = getattr(obs, "feedback", "")
            print(f"     → reward={reward:.2f} | {feedback[:100]}")

            time.sleep(0.3)  # rate-limit courtesy

    avg = total_reward / steps if steps > 0 else 0.0
    print(f"\n  ✅ {task_level.upper()} complete — steps={steps}, avg_score={avg:.3f}")
    return avg


def main():
    print(f"\n🚀 Email Triage Baseline")
    print(f"   model    : {MODEL_NAME}")
    print(f"   env      : {ENV_BASE_URL}")
    print(f"   running all 3 task levels...\n")

    results = {}
    for level in ["easy", "medium", "hard"]:
        try:
            score = run_task(level)
            results[level] = score
        except Exception as e:
            print(f"  ERROR on {level}: {e}")
            results[level] = 0.0

    print(f"\n{'='*60}")
    print("  BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    for level, score in results.items():
        bar = "█" * int(score * 20)
        print(f"  {level:8s}: {score:.3f}  {bar}")
    overall = sum(results.values()) / len(results)
    print(f"  {'overall':8s}: {overall:.3f}")
    print(f"{'='*60}\n")

    os.makedirs("outputs/evals", exist_ok=True)
    output = {
        "model": MODEL_NAME,
        "env_base_url": ENV_BASE_URL,
        "scores": results,
        "overall": overall,
    }
    with open("outputs/evals/baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("  📊 Results saved to outputs/evals/baseline_results.json")


if __name__ == "__main__":
    main()

