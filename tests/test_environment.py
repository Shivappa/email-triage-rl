"""
tests/test_environment.py — Unit tests for the Email Triage environment.

Tests:
  - reset() returns a valid TriageObservation
  - step() returns correct reward type and range
  - All 3 task levels complete without errors
  - Graders return scores in [0.0, 1.0]
  - Perfect action scores >= 0.9
  - Wrong action scores <= 0.5
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from email_triage_rl_hackathon.server.my_environment import EmailTriageEnvironment
from email_triage_rl_hackathon.models import TriageAction
from email_triage_rl_hackathon.server.graders import grade_easy, grade_medium, grade_hard
from email_triage_rl_hackathon.server.email_data import EASY_EMAILS, MEDIUM_EMAILS, HARD_EMAILS


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def perfect_action(email) -> TriageAction:
    """Return the ground-truth action for an email."""
    return TriageAction(
        category=email.gt_category,
        priority=email.gt_priority,
        department=email.gt_department,
        reply=" ".join(email.gt_reply_keywords) + " apologize refund investigate dear",
    )


def wrong_action() -> TriageAction:
    return TriageAction(
        category="spam",
        priority="low",
        department="support",
        reply="",
    )


# ─────────────────────────────────────────────
# Environment lifecycle tests
# ─────────────────────────────────────────────
class TestEnvironmentLifecycle:

    def test_reset_returns_valid_observation(self):
        env = EmailTriageEnvironment(task_level="easy")
        obs = env.reset()
        assert obs.email_id != "none"
        assert len(obs.subject) > 0
        assert obs.task_level == "easy"
        assert obs.done is False
        assert 0.0 <= obs.score <= 1.0

    def test_reset_with_seed_is_reproducible(self):
        env1 = EmailTriageEnvironment(task_level="easy")
        env2 = EmailTriageEnvironment(task_level="easy")
        obs1 = env1.reset(seed=42)
        obs2 = env2.reset(seed=42)
        assert obs1.email_id == obs2.email_id

    def test_state_returns_state_object(self):
        env = EmailTriageEnvironment(task_level="easy")
        env.reset()
        state = env.state
        assert state.episode_id is not None
        assert state.step_count == 0

    def test_step_increments_step_count(self):
        env = EmailTriageEnvironment(task_level="easy")
        obs = env.reset()
        action = TriageAction(category="spam", priority="low", department="support", reply="")
        env.step(action)
        assert env.state.step_count == 1

    def test_episode_ends_after_all_emails(self):
        for level in ["easy", "medium", "hard"]:
            env = EmailTriageEnvironment(task_level=level)
            obs = env.reset()
            action = wrong_action()
            done = False
            steps = 0
            while not done:
                result = env.step(action)
                done = result.done
                steps += 1
                assert steps < 20, "Episode did not terminate"
            assert done is True

    def test_reward_is_in_valid_range(self):
        env = EmailTriageEnvironment(task_level="easy")
        obs = env.reset()
        while not obs.done:
            action = TriageAction(category="billing", priority="medium",
                                  department="finance", reply="")
            obs = env.step(action)
            assert -1.0 <= obs.reward <= 1.0


# ─────────────────────────────────────────────
# Grader tests
# ─────────────────────────────────────────────
class TestGraders:

    def test_easy_perfect_score(self):
        email = EASY_EMAILS[1]  # billing email
        score, _ = grade_easy(email, email.gt_category, email.gt_priority, email.gt_department)
        assert score == 1.0

    def test_easy_all_wrong_score(self):
        email = EASY_EMAILS[0]  # spam email
        score, _ = grade_easy(email, "technical", "urgent", "engineering")
        assert score == 0.0

    def test_easy_partial_score(self):
        email = EASY_EMAILS[1]  # billing/high/finance
        score, _ = grade_easy(email, "billing", "low", "support")
        assert 0.0 < score < 1.0
        assert score == pytest.approx(0.40)  # only category correct

    def test_medium_no_reply_penalised(self):
        email = MEDIUM_EMAILS[0]
        score, feedback = grade_medium(
            email, email.gt_category, email.gt_priority, email.gt_department, ""
        )
        assert "too short" in feedback or "missing" in feedback
        assert score <= 0.5

    def test_medium_good_reply_scores_high(self):
        email = MEDIUM_EMAILS[0]
        reply = "Dear customer, we sincerely apologize for the double charge. We will investigate and refund within 24-48 hours."
        score, _ = grade_medium(
            email, email.gt_category, email.gt_priority, email.gt_department, reply
        )
        assert score >= 0.7

    def test_hard_short_reply_penalised(self):
        email = HARD_EMAILS[0]
        score, feedback = grade_hard(
            email, email.gt_category, email.gt_priority, email.gt_department, "ok"
        )
        # Classification correct = 0.40 max; reply too short adds nothing
        assert score <= 0.42  # no reply bonus beyond classification

    def test_hard_full_reply_scores_high(self):
        email = HARD_EMAILS[0]
        reply = (
            "Dear customer, we sincerely apologize for the unacceptable experience with "
            "account ACC-00421. I am personally escalating this to our senior finance manager "
            "right now. We will resolve this within 24 hours and a manager will contact you "
            "directly. All previous case numbers have been reviewed."
        )
        score, _ = grade_hard(
            email, email.gt_category, email.gt_priority, email.gt_department, reply
        )
        assert score >= 0.6

    def test_scores_always_between_0_and_1(self):
        for email in EASY_EMAILS + MEDIUM_EMAILS + HARD_EMAILS:
            if email.task_level == "easy":
                s, _ = grade_easy(email, "spam", "low", "support")
            elif email.task_level == "medium":
                s, _ = grade_medium(email, "spam", "low", "support", "")
            else:
                s, _ = grade_hard(email, "spam", "low", "support", "")
            assert 0.0 <= s <= 1.0, f"Score {s} out of range for {email.email_id}"


# ─────────────────────────────────────────────
# Full episode integration tests
# ─────────────────────────────────────────────
class TestFullEpisode:

    def test_perfect_easy_episode_scores_1(self):
        env = EmailTriageEnvironment(task_level="easy")
        obs = env.reset(seed=0)
        total = 0.0
        steps = 0
        while not obs.done:
            # Find current email's ground truth from email_data
            from email_triage_rl_hackathon.server.email_data import ALL_EMAILS
            email = ALL_EMAILS.get(obs.email_id)
            assert email is not None, f"Unknown email_id: {obs.email_id}"
            action = perfect_action(email)
            obs = env.step(action)
            total += obs.reward or 0.0
            steps += 1
        assert steps > 0
        avg = total / steps
        assert avg >= 0.9, f"Perfect easy episode should score ≥0.9, got {avg:.2f}"
