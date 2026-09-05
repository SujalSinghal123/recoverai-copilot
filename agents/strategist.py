"""
Strategist Agent
-----------------
Role: "Given WHY the payment failed, what's the best recovery action?"

Implements a lightweight contextual multi-armed bandit (epsilon-greedy over
per-context action-value estimates). Context = (failure_reason, customer
segment). Arms = recovery actions below. The bandit updates its value
estimates from observed recovery outcomes, so RecoverAI's strategy actually
*improves over time* instead of following static if/else rules -- this is
the "agentic learning loop" that differentiates it from a rules engine.

Rewards are simulated in this demo via `simulate_reward()` (would be
replaced by real recovery-outcome webhooks in production).
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple

ACTIONS = [
    "instant_retry_same_method",
    "retry_alternate_method",
    "delayed_retry_smart_window",
    "send_payment_link_whatsapp",
    "send_payment_link_email",
    "offer_pay_later",
    "escalate_to_human_agent",
]

# Prior domain knowledge: which actions are even plausible for a reason,
# so the bandit doesn't have to learn obviously bad pairings from scratch
# (e.g. don't "instant retry" a risk_engine_block).
ELIGIBLE_ACTIONS = {
    "insufficient_funds": ["delayed_retry_smart_window", "send_payment_link_whatsapp", "offer_pay_later"],
    "bank_server_timeout": ["instant_retry_same_method", "delayed_retry_smart_window", "retry_alternate_method"],
    "otp_expired": ["instant_retry_same_method", "retry_alternate_method"],
    "card_expired": ["send_payment_link_whatsapp", "send_payment_link_email", "retry_alternate_method"],
    "risk_engine_block": ["escalate_to_human_agent", "send_payment_link_email"],
    "network_drop": ["instant_retry_same_method", "delayed_retry_smart_window"],
    "wrong_cvv": ["instant_retry_same_method", "send_payment_link_whatsapp"],
    "daily_limit_exceeded": ["delayed_retry_smart_window", "retry_alternate_method", "send_payment_link_whatsapp"],
}


@dataclass
class ActionDecision:
    txn_id: str
    context: Tuple[str, str]
    chosen_action: str
    expected_value: float
    explored: bool
    reasoning: str


class StrategistAgent:
    def __init__(self, epsilon: float = 0.15):
        self.epsilon = epsilon
        # Q[(reason, segment)][action] = running average reward estimate
        self.Q: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
        self.counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.history = []

    def _candidates(self, reason: str):
        return ELIGIBLE_ACTIONS.get(reason, ACTIONS)

    def decide(self, txn_id: str, failure_reason: str, customer_segment: str) -> ActionDecision:
        context = (failure_reason, customer_segment)
        candidates = self._candidates(failure_reason)

        for a in candidates:
            self.Q[context].setdefault(a, 0.5)  # optimistic init

        explore = random.random() < self.epsilon
        if explore:
            action = random.choice(candidates)
            reasoning = (
                f"Exploring '{action}' for context {context} to keep learning "
                f"(epsilon={self.epsilon})."
            )
        else:
            action = max(candidates, key=lambda a: self.Q[context][a])
            reasoning = (
                f"Exploiting best-known action '{action}' for context {context} "
                f"(estimated recovery value={self.Q[context][action]:.2f})."
            )

        decision = ActionDecision(
            txn_id=txn_id,
            context=context,
            chosen_action=action,
            expected_value=round(self.Q[context][action], 4),
            explored=explore,
            reasoning=reasoning,
        )
        self.history.append(decision)
        return decision

    def update(self, decision: ActionDecision, reward: float):
        """reward: 1.0 if payment was recovered, 0.0 if not."""
        context, action = decision.context, decision.chosen_action
        self.counts[context][action] += 1
        n = self.counts[context][action]
        old_q = self.Q[context][action]
        # incremental running average
        self.Q[context][action] = old_q + (reward - old_q) / n

    def simulate_reward(self, decision: ActionDecision) -> float:
        """
        Simulates a recovery outcome for demo/training purposes.
        Encodes rough real-world intuition: e.g. instant retry works well for
        transient errors (otp/network/timeout), delayed retry works better for
        insufficient_funds, escalation works best for risk blocks, etc.
        """
        reason, _segment = decision.context
        action = decision.chosen_action
        base_rates = {
            ("bank_server_timeout", "instant_retry_same_method"): 0.55,
            ("network_drop", "instant_retry_same_method"): 0.6,
            ("otp_expired", "instant_retry_same_method"): 0.5,
            ("otp_expired", "retry_alternate_method"): 0.4,
            ("insufficient_funds", "delayed_retry_smart_window"): 0.45,
            ("insufficient_funds", "send_payment_link_whatsapp"): 0.35,
            ("insufficient_funds", "offer_pay_later"): 0.5,
            ("card_expired", "send_payment_link_whatsapp"): 0.4,
            ("card_expired", "send_payment_link_email"): 0.25,
            ("card_expired", "retry_alternate_method"): 0.3,
            ("risk_engine_block", "escalate_to_human_agent"): 0.5,
            ("risk_engine_block", "send_payment_link_email"): 0.2,
            ("wrong_cvv", "instant_retry_same_method"): 0.3,
            ("wrong_cvv", "send_payment_link_whatsapp"): 0.35,
            ("daily_limit_exceeded", "delayed_retry_smart_window"): 0.4,
            ("daily_limit_exceeded", "retry_alternate_method"): 0.45,
            ("daily_limit_exceeded", "send_payment_link_whatsapp"): 0.3,
        }
        p = base_rates.get((reason, action), 0.2)
        return 1.0 if random.random() < p else 0.0
