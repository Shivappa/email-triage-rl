"""
Inference Script — Email Triage RL Environment
===================================
MANDATORY environment variables (injected by hackathon validator):
    API_BASE_URL        The LiteLLM proxy endpoint (REQUIRED)
    API_KEY             The proxy API key (REQUIRED)
    MODEL_NAME          The model identifier (default: gpt-4o-mini)
    TASK_LEVEL          easy | medium | hard (default: easy)

STDOUT FORMAT (required by hackathon evaluator):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Usage (local testing only):
    # Against local server:
    API_KEY=hf_xxx API_BASE_URL=https://router.huggingface.co/v1 python inference.py
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from email_triage_rl_hackathon import EmailTriageEnv, TriageAction

# ── Config ────────────────────────────────────────────────────────────────────
# These are injected by the hackathon validator. Read them at call time inside main()
# so they are resolved AFTER the validator sets them — never use a module-level default
# that could silently bypass the proxy.
MODEL_NAME       = os.environ.get("MODEL_NAME", "gpt-4o-mini")
# ENV_BASE_URL: validator injects this pointing to the running environment server.
# Fallback to the HF Space URL for local testing without ENV_BASE_URL set.
ENV_BASE_URL     = os.environ.get("ENV_BASE_URL", "https://Shivacode-rl-hackathon.hf.space")
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME", "")
TASK_LEVEL       = os.environ.get("TASK_LEVEL", "easy")   # easy | medium | hard
BENCHMARK        = "email_triage_rl_hackathon"
MAX_STEPS        = 15
TEMPERATURE      = 0.0
MAX_TOKENS       = 512
SUCCESS_SCORE_THRESHOLD = 0.5

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


# ── Main episode loop ─────────────────────────────────────────────────────────
async def main() -> None:
    # Read API credentials directly from environment — these are injected by the
    # hackathon validator and must NOT be overridden by any .env file or default.
    api_key      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    api_base_url = os.environ.get("API_BASE_URL")  # NO default — must be injected by validator

    if not api_key:
        raise EnvironmentError(
            "API_KEY is not set. The hackathon validator injects API_KEY automatically.\n"
            "For local testing: export API_KEY=hf_your_token_here"
        )
    if not api_base_url:
        # Fall back for local testing only; validator always injects this
        api_base_url = "https://router.huggingface.co/v1"
        print(f"[DEBUG] API_BASE_URL not set — using fallback: {api_base_url}", flush=True)

    print(f"[DEBUG] API_BASE_URL={api_base_url}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)

    # Create OpenAI client pointed at the validator's LiteLLM proxy
    client = OpenAI(api_key=api_key, base_url=api_base_url)

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

        # Compute final score: average reward, clamped strictly to (0.01, 0.99)
        # Validator requires score strictly between 0 and 1 (not 0.0, not 1.0)
        score   = sum(rewards) / len(rewards) if rewards else 0.0
        score   = min(max(score, 0.01), 0.99)
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
