"""
Communicator Agent
-------------------
Role: "Turn the Strategist's decision into an actual customer-facing
message/action."

Template-based NLG by default (fast, deterministic, free, works offline --
important for a demo). If an ANTHROPIC_API_KEY is present in the
environment, it will instead call the Claude API to generate a more natural,
context-aware message. This pluggable design is intentional: it shows
graceful degradation and a clear upgrade path to a full LLM-driven agent
in production.
"""

import os
import random
from dataclasses import dataclass


TEMPLATES = {
    "instant_retry_same_method": [
        "Hi {name}, it looks like your last payment of Rs.{amount} didn't go through due to a temporary glitch. We're retrying it now — no action needed!",
    ],
    "retry_alternate_method": [
        "Hi {name}, your payment of Rs.{amount} couldn't be completed. Want to try a different payment method? Tap here to retry: {link}",
    ],
    "delayed_retry_smart_window": [
        "Hi {name}, we'll automatically retry your Rs.{amount} payment in a few hours at a time it's more likely to succeed. Sit back — we've got this.",
    ],
    "send_payment_link_whatsapp": [
        "Hi {name}, your order is saved! Complete your Rs.{amount} payment anytime here: {link} (valid for 24 hrs)",
    ],
    "send_payment_link_email": [
        "Hi {name}, we noticed your payment of Rs.{amount} was interrupted. Here's a secure link to complete it: {link}",
    ],
    "offer_pay_later": [
        "Hi {name}, having trouble paying Rs.{amount} right now? You can split it into easy installments — check eligibility here: {link}",
    ],
    "escalate_to_human_agent": [
        "Hi {name}, your payment of Rs.{amount} was flagged for a routine security check. Our team will reach out shortly to help you complete it safely.",
    ],
}


@dataclass
class Message:
    txn_id: str
    channel: str
    body: str
    generated_by: str


class CommunicatorAgent:
    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    def _channel_for(self, action: str) -> str:
        if "whatsapp" in action:
            return "whatsapp"
        if "email" in action:
            return "email"
        if "human_agent" in action:
            return "internal_queue"
        return "sms"

    def compose(self, txn_id: str, action: str, customer_name: str, amount: float, link: str = "https://rzp.io/recover/xxxx") -> Message:
        if self.api_key:
            return self._compose_via_llm(txn_id, action, customer_name, amount, link)
        return self._compose_via_template(txn_id, action, customer_name, amount, link)

    def _compose_via_template(self, txn_id, action, customer_name, amount, link) -> Message:
        template = random.choice(TEMPLATES.get(action, TEMPLATES["send_payment_link_whatsapp"]))
        body = template.format(name=customer_name, amount=f"{amount:,.2f}", link=link)
        return Message(
            txn_id=txn_id,
            channel=self._channel_for(action),
            body=body,
            generated_by="template_engine",
        )

    def _compose_via_llm(self, txn_id, action, customer_name, amount, link) -> Message:
        # Kept intentionally simple / optional -- graceful fallback if the
        # API call fails for any reason (offline demo env, quota, etc).
        try:
            import urllib.request
            import json as _json

            prompt = (
                f"Write a short, warm, 2-sentence payment-recovery message for a customer "
                f"named {customer_name} whose Rs.{amount:.2f} payment failed. "
                f"Recovery action to take: {action}. Include this link naturally: {link}. "
                f"Tone: helpful, concise, not pushy."
            )
            payload = _json.dumps(
                {
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 150,
                    "messages": [{"role": "user", "content": prompt}],
                }
            ).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = _json.loads(resp.read())
                body = "".join(
                    block.get("text", "") for block in data.get("content", [])
                ).strip()
                if body:
                    return Message(txn_id, self._channel_for(action), body, "claude-sonnet-4-6")
        except Exception:
            pass
        return self._compose_via_template(txn_id, action, customer_name, amount, link)
