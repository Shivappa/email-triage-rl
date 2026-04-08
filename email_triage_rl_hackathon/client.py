"""
client.py — EnvClient for the Email Triage environment (client-side).
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TriageAction, TriageObservation


class EmailTriageEnv(EnvClient[TriageAction, TriageObservation, State]):
    """
    Client for the Email Triage Environment.

    Usage (sync):
        with EmailTriageEnv(base_url="http://localhost:8000").sync() as env:
            obs = env.reset()
            print(obs.subject)
            result = env.step(TriageAction(
                category="billing", priority="high",
                department="finance", reply="Dear customer..."
            ))
            print(result.reward)
    """

    def _step_payload(self, action: TriageAction) -> Dict:
        return {
            "category":   action.category,
            "priority":   action.priority,
            "department": action.department,
            "reply":      action.reply,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TriageObservation]:
        obs_data = payload.get("observation", {})
        observation = TriageObservation(
            email_id=obs_data.get("email_id", ""),
            subject=obs_data.get("subject", ""),
            body=obs_data.get("body", ""),
            sender=obs_data.get("sender", ""),
            task_level=obs_data.get("task_level", "easy"),
            feedback=obs_data.get("feedback", ""),
            score=obs_data.get("score", 0.0),
            emails_remaining=obs_data.get("emails_remaining", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
