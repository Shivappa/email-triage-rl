"""
graders.py — Deterministic graders for the three task levels.

Each grader returns a score in [0.0, 1.0] and a feedback string.

Scoring rubrics:
  EASY   (0.0-1.0): correct category + priority + department
  MEDIUM (0.0-1.0): classification score + reply quality score (50/50 split)
  HARD   (0.0-1.0): classification score + reply depth/accuracy score (40/60 split)
"""

from typing import Tuple
from email_triage_rl_hackathon.server.email_data import Email


def grade_easy(email: Email, category: str, priority: str, department: str) -> Tuple[float, str]:
    """
    Easy task grader — classification only, no reply needed.
    Score breakdown:
      category   correct → +0.40
      priority   correct → +0.30
      department correct → +0.30 
    """
    score = 0.0
    feedback_parts = []

    if category == email.gt_category:
        score += 0.40
        feedback_parts.append(f"✅ Category '{category}' is correct.")
    else:
        feedback_parts.append(
            f"❌ Category '{category}' is wrong (expected: '{email.gt_category}')."
        )

    if priority == email.gt_priority:
        score += 0.30
        feedback_parts.append(f"✅ Priority '{priority}' is correct.")
    else:
        feedback_parts.append(
            f"❌ Priority '{priority}' is wrong (expected: '{email.gt_priority}')."
        )

    if department == email.gt_department:
        score += 0.30
        feedback_parts.append(f"✅ Department '{department}' is correct.")
    else:
        feedback_parts.append(
            f"❌ Department '{department}' is wrong (expected: '{email.gt_department}')."
        )

    return round(score, 2), " ".join(feedback_parts)


def grade_medium(
    email: Email,
    category: str,
    priority: str,
    department: str,
    reply: str,
) -> Tuple[float, str]:
    """
    Medium task grader — classification + reply quality.
    Score breakdown:
      classification (50%):
        category   correct → +0.20
        priority   correct → +0.15
        department correct → +0.15
      reply quality (50%):
        non-empty reply            → +0.10
        professional tone          → +0.10
        each keyword matched (×n)  → up to +0.30 total
    """
    score = 0.0
    feedback_parts = []

    # — Classification —
    if category == email.gt_category:
        score += 0.20
        feedback_parts.append(f"✅ Category '{category}' correct.")
    else:
        feedback_parts.append(f"❌ Category wrong (expected '{email.gt_category}').")

    if priority == email.gt_priority:
        score += 0.15
        feedback_parts.append(f"✅ Priority '{priority}' correct.")
    else:
        feedback_parts.append(f"❌ Priority wrong (expected '{email.gt_priority}').")

    if department == email.gt_department:
        score += 0.15
        feedback_parts.append(f"✅ Department '{department}' correct.")
    else:
        feedback_parts.append(f"❌ Department wrong (expected '{email.gt_department}').")

    # — Reply quality —
    reply_lower = reply.lower().strip()

    if len(reply_lower) < 20:
        feedback_parts.append("❌ Reply too short or missing.")
    else:
        score += 0.10
        feedback_parts.append("✅ Reply provided.")

        # Professional tone markers
        tone_markers = ["dear", "hello", "hi", "thank", "please", "sincerely",
                        "regards", "apologi", "we will", "we are"]
        if any(m in reply_lower for m in tone_markers):
            score += 0.10
            feedback_parts.append("✅ Professional tone detected.")
        else:
            feedback_parts.append("❌ Reply lacks professional tone.")

        # Keyword matching
        if email.gt_reply_keywords:
            matched = [kw for kw in email.gt_reply_keywords if kw.lower() in reply_lower]
            kw_score = (len(matched) / len(email.gt_reply_keywords)) * 0.30
            score += kw_score
            if matched:
                feedback_parts.append(
                    f"✅ Reply keywords matched ({len(matched)}/{len(email.gt_reply_keywords)}): "
                    f"{', '.join(matched)}."
                )
            else:
                feedback_parts.append(
                    f"❌ No required keywords found. Expected some of: "
                    f"{', '.join(email.gt_reply_keywords)}."
                )

    return round(min(score, 1.0), 2), " ".join(feedback_parts)


def grade_hard(
    email: Email,
    category: str,
    priority: str,
    department: str,
    reply: str,
) -> Tuple[float, str]:
    """
    Hard task grader — classification + deep reply accuracy.
    Score breakdown:
      classification (40%):
        category   correct → +0.15
        priority   correct → +0.13
        department correct → +0.12
      reply quality (60%):
        non-empty (≥50 chars)      → +0.05
        professional tone          → +0.05
        length ≥ 100 chars         → +0.05
        keyword coverage           → up to 0.45 (proportional)
    """
    score = 0.0
    feedback_parts = []

    # — Classification —
    if category == email.gt_category:
        score += 0.15
        feedback_parts.append(f"✅ Category '{category}' correct.")
    else:
        feedback_parts.append(f"❌ Category wrong (expected '{email.gt_category}').")

    if priority == email.gt_priority:
        score += 0.13
        feedback_parts.append(f"✅ Priority '{priority}' correct.")
    else:
        feedback_parts.append(f"❌ Priority wrong (expected '{email.gt_priority}').")

    if department == email.gt_department:
        score += 0.12
        feedback_parts.append(f"✅ Department '{department}' correct.")
    else:
        feedback_parts.append(f"❌ Department wrong (expected '{email.gt_department}').")

    # — Reply quality —
    reply_lower = reply.lower().strip()

    if len(reply_lower) < 50:
        feedback_parts.append("❌ Reply too short (need ≥50 chars for hard tasks).")
    else:
        score += 0.05
        feedback_parts.append("✅ Reply meets minimum length.")

        if len(reply_lower) >= 100:
            score += 0.05
            feedback_parts.append("✅ Reply has good length (≥100 chars).")

        tone_markers = ["dear", "hello", "hi", "thank", "please", "sincerely",
                        "regards", "apologi", "we will", "we are", "i understand"]
        if any(m in reply_lower for m in tone_markers):
            score += 0.05
            feedback_parts.append("✅ Professional tone detected.")
        else:
            feedback_parts.append("❌ Reply lacks professional tone.")

        if email.gt_reply_keywords:
            matched = [kw for kw in email.gt_reply_keywords if kw.lower() in reply_lower]
            kw_score = (len(matched) / len(email.gt_reply_keywords)) * 0.45
            score += kw_score
            feedback_parts.append(
                f"{'✅' if matched else '❌'} Keywords matched "
                f"({len(matched)}/{len(email.gt_reply_keywords)}): "
                f"{', '.join(matched) if matched else 'none'}."
            )

    return round(min(score, 1.0), 2), " ".join(feedback_parts)


def grade(email: Email, category: str, priority: str, department: str, reply: str) -> Tuple[float, str]:
    """Route to the correct grader based on task_level."""
    if email.task_level == "easy":
        return grade_easy(email, category, priority, department)
    elif email.task_level == "medium":
        return grade_medium(email, category, priority, department, reply)
    else:
        return grade_hard(email, category, priority, department, reply)
