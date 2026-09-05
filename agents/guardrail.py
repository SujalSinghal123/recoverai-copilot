"""
Risk Guardrail Agent
----------------------
Role: "Should we actually let this recovery action run?"

This is the safety layer between the Strategist's decision and the
Communicator's execution. It exists because an adaptive, exploring policy
(the Strategist's bandit) is exactly the kind of component you do NOT want
making unsupervised contact with a customer when a transaction looks
fraudulent -- an automated "here's a payment link, please pay again" is the
wrong move on a high-risk transaction, no matter how promising the bandit
thinks that action is.

The Guardrail scores every transaction's fraud risk (`fraud_score`, already
present on the incoming transaction in this demo -- in production this would
come from Razorpay's real risk engine) and, above a threshold, VETOES any
automated/self-serve recovery action and forces escalation to a human agent
instead. It never blocks escalation itself, and it never touches the
Strategist's learned values directly -- a vetoed action is treated as
"never executed" so the bandit isn't corrupted with a fake outcome.

This directly bridges Track 3 (Revenue Recovery) into Track 2 (AI Risk
Manager), as called out in the roadmap.
"""

from dataclasses import dataclass
from typing import Optional

# Actions considered "self-serve" / automated -- the ones a fraud ring could
# exploit if left on autopilot. Anything not in this set (i.e. already
# human-in-the-loop) is left alone by the guardrail.
AUTOMATED_ACTIONS = {
    "instant_retry_same_method",
    "retry_alternate_method",
    "delayed_retry_smart_window",
    "send_payment_link_whatsapp",
    "send_payment_link_email",
    "offer_pay_later",
}

FALLBACK_ACTION = "escalate_to_human_agent"


@dataclass
class GuardrailReview:
    txn_id: str
    fraud_score: float
    approved: bool
    final_action: str
    reasoning: str


class RiskGuardrailAgent:
    def __init__(self, veto_threshold: float = 0.65):
        self.veto_threshold = veto_threshold
        self.veto_count = 0
        self.review_count = 0

    def review(self, txn_id: str, fraud_score: float, proposed_action: str) -> GuardrailReview:
        self.review_count += 1
        high_risk = fraud_score >= self.veto_threshold
        is_automated = proposed_action in AUTOMATED_ACTIONS

        if high_risk and is_automated:
            self.veto_count += 1
            return GuardrailReview(
                txn_id=txn_id,
                fraud_score=round(fraud_score, 3),
                approved=False,
                final_action=FALLBACK_ACTION,
                reasoning=(
                    f"VETOED '{proposed_action}' — fraud_score {fraud_score:.2f} is at/above the "
                    f"{self.veto_threshold:.2f} guardrail threshold. Routing to a human agent instead "
                    f"of letting an automated recovery message reach a high-risk transaction."
                ),
            )

        return GuardrailReview(
            txn_id=txn_id,
            fraud_score=round(fraud_score, 3),
            approved=True,
            final_action=proposed_action,
            reasoning=(
                f"Approved '{proposed_action}' — fraud_score {fraud_score:.2f} is below the "
                f"{self.veto_threshold:.2f} guardrail threshold."
            ),
        )
