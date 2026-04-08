"""
models.py — Typed Action / Observation models for the Email Triage environment.

Real-world task: An AI agent reads incoming emails and must:
  1. Classify the email category
  2. Assign a priority level
  3. Route to the correct department + draft an appropriate reply

Action Space:
  - category   : spam | billing | technical | hr | general
  - priority   : low | medium | high | urgent
  - department : support | finance | engineering | hr | management
  - reply      : free-text reply draft (required for medium/hard tasks)

Observation Space:
  - email_id       : unique ID for the current email
  - subject        : email subject line
  - body           : email body text
  - sender         : sender address
  - task_level     : easy | medium | hard
  - feedback       : natural-language feedback on the last action
  - score          : 0.0-1.0 cumulative task score
  - done           : episode finished flag
  - reward         : step reward
  - emails_remaining: how many emails are left in the episode
"""

from typing import Literal, Optional
from openenv.core.env_server.types import Action, Observation

Category   = Literal["spam", "billing", "technical", "hr", "general"]
Priority   = Literal["low", "medium", "high", "urgent"]
Department = Literal["support", "finance", "engineering", "hr", "management"]


class TriageAction(Action):
    """Agent's triage decision for one email."""
    category:   Category
    priority:   Priority
    department: Department
    reply:      str = ""          # Required for medium / hard tasks


class TriageObservation(Observation):
    """What the agent sees after each step."""
    email_id:          str
    subject:           str
    body:              str
    sender:            str
    task_level:        Literal["easy", "medium", "hard"]
    feedback:          str   = ""
    score:             float = 0.0
    emails_remaining:  int   = 0
