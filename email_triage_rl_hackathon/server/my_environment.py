"""
email_triage_environment.py — Email Triage RL Environment (server-side).

Real-world task: triage incoming emails by category, priority, department,
and draft appropriate replies. Three difficulty levels with deterministic graders.

  reset(seed)  → start a new episode, returns first email as Observation
  step(action) → grade the triage decision, returns next email + reward
  state        → current episode metadata (episode_id, step_count)
"""

import random
from uuid import uuid4
from typing import Optional, Literal

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TriageAction, TriageObservation
    from .email_data import EMAILS_BY_LEVEL, Email
    from .graders import grade
except ImportError:
    from email_triage_rl_hackathon.models import TriageAction, TriageObservation
    from email_triage_rl_hackathon.server.email_data import EMAILS_BY_LEVEL, Email
    from email_triage_rl_hackathon.server.graders import grade


TaskLevel = Literal["easy", "medium", "hard"]


class EmailTriageEnvironment(Environment):
    """
    Email Triage Environment — a real-world RL task.

    An AI agent receives a queue of emails and must triage each one by:
      - Classifying it (spam / billing / technical / hr / general)
      - Assigning a priority (low / medium / high / urgent)
      - Routing to the correct department
      - (medium/hard) Drafting a professional reply

    Three task levels with increasing difficulty:
      easy   — classification only (5 emails)
      medium — classification + short reply (3 emails)
      hard   — classification + detailed accurate reply (3 emails)

    Rewards are partial and continuous (0.0–1.0 per email).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_level: TaskLevel = "easy"):
        self._task_level: TaskLevel = task_level
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._queue: list[Email] = []
        self._current_email: Optional[Email] = None
        self._cumulative_score: float = 0.0
        self._rng = random.Random()

    # ─────────────────────────────────────────────
    # reset
    # ─────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None, task_level: Optional[TaskLevel] = None) -> TriageObservation:
        if task_level:
            self._task_level = task_level

        self._rng = random.Random(seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._cumulative_score = 0.0

        emails = list(EMAILS_BY_LEVEL[self._task_level])
        self._rng.shuffle(emails)
        self._queue = emails
        self._current_email = self._queue.pop(0)

        return self._make_observation(
            feedback="New episode started. Please triage the email below.",
            reward=0.01,
            done=False,
        )

    # ─────────────────────────────────────────────
    # step
    # ─────────────────────────────────────────────
    def step(self, action: TriageAction) -> TriageObservation:  # type: ignore[override]
        if self._current_email is None:
            return TriageObservation(
                email_id="none",
                subject="",
                body="Episode not started. Call reset() first.",
                sender="",
                task_level=self._task_level,
                feedback="Call reset() to start.",
                score=0.01,
                emails_remaining=0,
                done=True,
                reward=0.01,
            )

        self._state.step_count += 1

        # Grade the action
        step_score, feedback = grade(
            email=self._current_email,
            category=action.category,
            priority=action.priority,
            department=action.department,
            reply=action.reply,
        )
        self._cumulative_score += step_score

        # Partial reward shaping: penalise missing reply on medium/hard
        reward = step_score
        if self._task_level in ("medium", "hard") and len(action.reply.strip()) < 20:
            reward = max(0.01, reward - 0.1)  # penalise empty reply, never below 0.01

        # Clamp reward to strictly (0.01, 0.99) — validator requires score != 0.0 and != 1.0
        reward = min(max(reward, 0.01), 0.99)

        # Advance to next email
        if self._queue:
            self._current_email = self._queue.pop(0)
            done = False
        else:
            self._current_email = None
            done = True

        avg_score = self._cumulative_score / self._state.step_count

        return self._make_observation(
            feedback=f"[Email graded — score: {step_score:.2f}] {feedback}",
            reward=reward,
            done=done,
            avg_score=avg_score,
        )

    # ─────────────────────────────────────────────
    # state (property)
    # ─────────────────────────────────────────────
    @property
    def state(self) -> State:
        return self._state

    # ─────────────────────────────────────────────
    # helpers
    # ─────────────────────────────────────────────
    def _make_observation(
        self,
        feedback: str,
        reward: float,
        done: bool,
        avg_score: float = 0.0,
    ) -> TriageObservation:
        email = self._current_email
        if email is None:
            return TriageObservation(
                email_id="none",
                subject="[Episode Complete]",
                body=f"All emails triaged. Average score: {avg_score:.2f}",
                sender="system",
                task_level=self._task_level,
                feedback=feedback,
                score=avg_score,
                emails_remaining=0,
                done=True,
                reward=reward,
            )
        return TriageObservation(
            email_id=email.email_id,
            subject=email.subject,
            body=email.body,
            sender=email.sender,
            task_level=self._task_level,
            feedback=feedback,
            score=avg_score,
            emails_remaining=len(self._queue),
            done=done,
            reward=reward,
        )

