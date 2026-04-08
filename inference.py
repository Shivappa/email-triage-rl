"""
Inference Script — Email Triage RL Environment
===================================
MANDATORY environment variables:
    API_BASE_URL        The API endpoint for the LLM
                        (default: https://router.huggingface.co/v1)
    MODEL_NAME          The model identifier
                        (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN            Your Hugging Face / API key
    LOCAL_IMAGE_NAME    Docker image name (if using from_docker_image())

STDOUT FORMAT (required by hackathon evaluator):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Usage:
    # Against local server (start first: uvicorn email_triage_rl_hackathon.server.app:app --port 8000)
    HF_TOKEN=hf_xxx python inference.py

    # Against deployed HF Space
    ENV_BASE_URL=https://Shivacode-rl-hackathon.hf.space HF_TOKEN=hf_xxx python inference.py

    # With Docker image
    LOCAL_IMAGE_NAME=email-triage-rl-hackathon:latest HF_TOKEN=hf_xxx python inference.py

    # Choose task level
    TASK_LEVEL=hard HF_TOKEN=hf_xxx python inference.py
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

# Auto-load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from email_triage_rl_hackathon import EmailTriageEnv, TriageAction

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY          = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL     = os.getenv("ENV_BASE_URL", "http://localhost:8000")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
TASK_LEVEL       = os.getenv("TASK_LEVEL", "easy")   # easy | medium | hard
BENCHMARK        = "email_triage_rl_hackathon"
MAX_STEPS        = 15    # more than enough for all 3 levels (max 5 emails)
TEMPERATURE      = 0.0   # deterministic for reproducibility
MAX_TOKENS       = 300
SUCCESS_SCORE_THRESHOLD = 0.5  # episode is "successful" if avg score >= 0.5

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert email triage agent working for a company.
    For each email you receive, respond with a JSON object with EXACTLY these fields:
    {
      "category":   one of ["spam", "billing", "technical", "hr", "general"],
      "priority":   one of ["low", "medium", "high", "urgent"],
      "department": one of ["support", "finance", "engineering", "hr", "management"],
      "reply":      "your professional reply draft (empty string for easy tasks)"
    }

    Routing rules:
    - spam               → category=spam,      priority=low,    department=support
    - payment/invoice    → category=billing,   priority=high,   department=finance
    - bugs/login/API     → category=technical, priority=urgent, department=engineering
    - leave/HR/complaint → category=hr,        priority=urgent, department=hr
    - board/executive    → category=hr,        priority=urgent, department=management
    - general/supplies   → category=general,   priority=low,    department=support

    For medium/hard tasks: write a professional reply (minimum 2 sentences).
    For urgent issues: always acknowledge urgency and give a timeframe in the reply.
    Respond ONLY with the JSON object, no other text.
""").strip()


# ── Logging helpers (required stdout format) ──────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM call ─────────────────────────────────────────────────────────────────
def build_user_prompt(obs, step: int, history: List[str]) -> str:
    history_block = "\n".join(history[-3:]) if history else "None"
    return textwrap.dedent(f"""
        Step: {step}
        Task level: {obs.task_level}
        Emails remaining after this: {obs.emails_remaining}

        --- EMAIL ---
        From:    {obs.sender}
        Subject: {obs.subject}

        {obs.body}
        --- END EMAIL ---

        Previous steps:
        {history_block}

        {"Note: Write a professional reply (reply field required)." if obs.task_level in ("medium", "hard") else "Note: Reply field can be empty string for easy tasks."}
    """).strip()


def get_model_action(client: OpenAI, obs, step: int, history: List[str]) -> TriageAction:
    """Call the LLM and parse the JSON response into a TriageAction."""
    user_prompt = build_user_prompt(obs, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        data = json.loads(content)
        return TriageAction(**data)

    except Exception as exc:
        print(f"[DEBUG] Model parse error: {exc}", flush=True)
        # Safe fallback action
        return TriageAction(
            category="general",
            priority="low",
            department="support",
            reply="",
        )


# ── Main episode loop ─────────────────────────────────────────────────────────
async def main() -> None:
    if not API_KEY:
        raise EnvironmentError(
            "HF_TOKEN is not set. Add it to your .env file or export it:\n"
            "  export HF_TOKEN=hf_your_token_here"
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to environment — Docker image or running server
    if LOCAL_IMAGE_NAME:
        print(f"[DEBUG] Connecting via Docker image: {LOCAL_IMAGE_NAME}", flush=True)
        env = await EmailTriageEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        print(f"[DEBUG] Connecting to server: {ENV_BASE_URL}", flush=True)
        env = EmailTriageEnv(base_url=ENV_BASE_URL)

    history:     List[str]  = []
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=TASK_LEVEL, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset episode
        reset_result = await env.reset(seed=42, task_level=TASK_LEVEL)
        # reset() may return a StepResult or the observation directly
        obs = reset_result.observation if hasattr(reset_result, "observation") else reset_result

        for step in range(1, MAX_STEPS + 1):
            if obs.done or obs.email_id == "none":
                break

            # Get LLM decision
            action = get_model_action(client, obs, step, history)

            # Compact action string for logging (no newlines)
            action_str = (
                f"category={action.category},priority={action.priority},"
                f"department={action.department},"
                f"reply={repr(action.reply[:40]) if action.reply else repr('')}"
            )

            # Step the environment
            result = await env.step(action)

            reward      = result.reward or 0.0
            done        = result.done
            error       = None
            steps_taken = step
            obs         = result.observation

            rewards.append(reward)

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step} [{obs.email_id}]: "
                f"cat={action.category} pri={action.priority} "
                f"→ reward={reward:.2f} | {obs.feedback[:60]}"
            )

            if done:
                break

        # Compute final score: average reward across all steps, clamped to [0, 1]
        score   = sum(rewards) / len(rewards) if rewards else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
