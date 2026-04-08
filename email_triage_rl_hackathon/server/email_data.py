"""
email_data.py — Curated email dataset for the three task levels.

Each email has ground-truth labels used by the graders.
Fields:
  email_id, subject, body, sender,
  gt_category, gt_priority, gt_department,
  gt_reply_keywords  — list of words that MUST appear in a good reply
  task_level
"""

from typing import List
from dataclasses import dataclass, field


@dataclass
class Email:
    email_id:           str
    subject:            str
    body:               str
    sender:             str
    gt_category:        str
    gt_priority:        str
    gt_department:      str
    gt_reply_keywords:  List[str]
    task_level:         str


# ─────────────────────────────────────────────────────────────────
# EASY EMAILS  — category + priority + department (no reply needed)
# ─────────────────────────────────────────────────────────────────
EASY_EMAILS: List[Email] = [
    Email(
        email_id="E001",
        subject="Congratulations! You've won $1,000,000!",
        body=(
            "Dear Friend, You have been selected as the lucky winner of our "
            "international lottery. Click here to claim your prize immediately. "
            "Send us your bank details to process the transfer."
        ),
        sender="winner@totally-legit-lottery.ru",
        gt_category="spam",
        gt_priority="low",
        gt_department="support",
        gt_reply_keywords=[],
        task_level="easy",
    ),
    Email(
        email_id="E002",
        subject="Invoice #4821 — Payment Overdue",
        body=(
            "Hi, I noticed that invoice #4821 for $3,200 issued on March 1st "
            "has not been paid yet. The due date was March 15th. Could you "
            "please arrange payment or let me know if there's an issue?"
        ),
        sender="accounts@vendorcorp.com",
        gt_category="billing",
        gt_priority="high",
        gt_department="finance",
        gt_reply_keywords=[],
        task_level="easy",
    ),
    Email(
        email_id="E003",
        subject="Cannot login to the dashboard",
        body=(
            "Hi support team, I've been unable to log into the dashboard since "
            "this morning. I've tried resetting my password twice but still get "
            "an 'invalid credentials' error. My username is john.doe@company.com."
        ),
        sender="john.doe@company.com",
        gt_category="technical",
        gt_priority="high",
        gt_department="engineering",
        gt_reply_keywords=[],
        task_level="easy",
    ),
    Email(
        email_id="E004",
        subject="Annual leave request — 3 days",
        body=(
            "Dear HR, I would like to request 3 days of annual leave from "
            "April 14 to April 16. I have 12 days remaining in my balance. "
            "Please let me know if this is approved. Thanks, Sarah."
        ),
        sender="sarah.jones@company.com",
        gt_category="hr",
        gt_priority="medium",
        gt_department="hr",
        gt_reply_keywords=[],
        task_level="easy",
    ),
    Email(
        email_id="E005",
        subject="Office supplies order",
        body=(
            "Hi, we're running low on printer paper and pens in office 3B. "
            "Could someone arrange a restock? We typically order from OfficeMax. "
            "Nothing urgent, just flagging before we run out."
        ),
        sender="mike.brown@company.com",
        gt_category="general",
        gt_priority="low",
        gt_department="support",
        gt_reply_keywords=[],
        task_level="easy",
    ),
]

# ─────────────────────────────────────────────────────────────────
# MEDIUM EMAILS  — classification + short professional reply
# ─────────────────────────────────────────────────────────────────
MEDIUM_EMAILS: List[Email] = [
    Email(
        email_id="M001",
        subject="Double charge on my account",
        body=(
            "Hello, I was charged twice for my subscription on April 3rd — "
            "two charges of $49.99 each appear on my credit card statement. "
            "Order numbers: ORD-8821 and ORD-8822. Please refund the duplicate "
            "charge as soon as possible. My account email is lisa@example.com."
        ),
        sender="lisa@example.com",
        gt_category="billing",
        gt_priority="urgent",
        gt_department="finance",
        gt_reply_keywords=["apologize", "refund", "investigate", "24", "48"],
        task_level="medium",
    ),
    Email(
        email_id="M002",
        subject="API rate limit errors in production",
        body=(
            "Hi engineering team, we are hitting 429 rate limit errors on the "
            "/api/v2/data endpoint in production since 14:00 UTC today. "
            "Our integration handles ~500 requests/minute which was previously "
            "fine. No changes were made on our side. This is impacting live "
            "customers. Please investigate urgently."
        ),
        sender="devops@clientbusiness.com",
        gt_category="technical",
        gt_priority="urgent",
        gt_department="engineering",
        gt_reply_keywords=["investigate", "team", "update", "priority", "contact"],
        task_level="medium",
    ),
    Email(
        email_id="M003",
        subject="Harassment complaint — requires immediate attention",
        body=(
            "I need to formally report that I have been experiencing workplace "
            "harassment from a colleague in my department. This has been ongoing "
            "for 3 weeks. I have documented instances with dates. I would like "
            "to speak with someone in HR confidentially as soon as possible."
        ),
        sender="anonymous.employee@company.com",
        gt_category="hr",
        gt_priority="urgent",
        gt_department="hr",
        gt_reply_keywords=["seriously", "confidential", "contact", "meet", "support"],
        task_level="medium",
    ),
]

# ─────────────────────────────────────────────────────────────────
# HARD EMAILS  — multi-signal triage + detailed, accurate reply
# ─────────────────────────────────────────────────────────────────
HARD_EMAILS: List[Email] = [
    Email(
        email_id="H001",
        subject="Re: Re: Re: Billing issue — still unresolved after 3 weeks",
        body=(
            "I have now contacted your support team FOUR TIMES about the incorrect "
            "charge of $1,250 on my enterprise account (ACC-00421). Each time I was "
            "told 'it will be resolved within 3 business days'. It has been 3 weeks. "
            "I have email threads, case numbers CS-1142, CS-1198, CS-1267 and CS-1301. "
            "If this is not resolved by end of week I will be filing a chargeback and "
            "escalating to my legal team. This is completely unacceptable."
        ),
        sender="cto@enterprise-client.com",
        gt_category="billing",
        gt_priority="urgent",
        gt_department="finance",
        gt_reply_keywords=[
            "apologize", "personally", "escalate", "manager", "resolve",
            "24", "hours", "case", "ACC-00421"
        ],
        task_level="hard",
    ),
    Email(
        email_id="H002",
        subject="Security incident — possible data breach",
        body=(
            "Our security team has detected unusual outbound traffic from your "
            "platform at 02:30 UTC today. We are seeing repeated requests to "
            "external IPs originating from our tenant environment (tenant ID: "
            "T-9942). Logs indicate approximately 50,000 records may have been "
            "accessed. We require immediate incident response. Our CISO is "
            "available on +1-555-0199. Please treat this as P0."
        ),
        sender="security@enterprise-partner.com",
        gt_category="technical",
        gt_priority="urgent",
        gt_department="engineering",
        gt_reply_keywords=[
            "immediately", "incident", "team", "isolate", "investigate",
            "contact", "CISO", "P0", "hour"
        ],
        task_level="hard",
    ),
    Email(
        email_id="H003",
        subject="Board-level complaint: employee misconduct + financial irregularities",
        body=(
            "I am writing on behalf of two board members who wish to remain "
            "anonymous at this stage. We have evidence of a senior manager in "
            "the finance department approving unauthorized expenses totalling "
            "over $85,000 over the past six months. Additionally, there are "
            "allegations of related-party transactions that were not disclosed. "
            "This requires a confidential internal audit and independent HR "
            "investigation. We request a response to the board within 48 hours "
            "outlining the steps being taken."
        ),
        sender="board.office@company.com",
        gt_category="hr",
        gt_priority="urgent",
        gt_department="management",
        gt_reply_keywords=[
            "acknowledge", "confidential", "audit", "investigate",
            "48", "hours", "board", "seriously", "independent"
        ],
        task_level="hard",
    ),
]

ALL_EMAILS = {e.email_id: e for e in EASY_EMAILS + MEDIUM_EMAILS + HARD_EMAILS}
EMAILS_BY_LEVEL = {
    "easy":   EASY_EMAILS,
    "medium": MEDIUM_EMAILS,
    "hard":   HARD_EMAILS,
}
