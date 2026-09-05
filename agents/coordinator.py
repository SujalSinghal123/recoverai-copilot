"""
Coordinator
------------
Orchestrates Diagnostician -> Strategist -> Communicator for a batch of
failed transactions, logs the full reasoning trace of every agent (for
explainability/audit -- important in a payments/fintech context), and
computes the business-facing metric that matters most: INR revenue
recovered.
"""

import os
import random
from collections import defaultdict
from dataclasses import asdict

import pandas as pd

from agents.diagnostician import DiagnosticianAgent
from agents.strategist import StrategistAgent
from agents.guardrail import RiskGuardrailAgent
from agents.communicator import CommunicatorAgent

FIRST_NAMES = ["Aarav", "Priya", "Rohan", "Ishita", "Kabir", "Ananya", "Vivaan", "Sara", "Aditya", "Neha"]


class RecoverAICoordinator:
    def __init__(self):
        self.diagnostician = DiagnosticianAgent()
        self.strategist = StrategistAgent()
        self.guardrail = RiskGuardrailAgent()
        self.communicator = CommunicatorAgent()
        self.trace = []
        self.metrics = {}

    def train(self, csv_path: str):
        self.metrics["diagnostician"] = self.diagnostician.train(csv_path)
        return self.metrics["diagnostician"]

    def run_batch(self, csv_path: str, n: int = 200):
        df = pd.read_csv(csv_path).sample(n=min(n, len(pd.read_csv(csv_path))), random_state=random.randint(0, 9999))
        df["is_weekend"] = df["is_weekend"].astype(int)

        total_amount = 0.0
        recovered_amount = 0.0
        recovered_count = 0
        held_count = 0
        held_amount = 0.0
        by_reason = defaultdict(lambda: {"recovered_amount": 0.0, "recovered_count": 0, "seen": 0})

        for _, row in df.iterrows():
            txn = row.to_dict()
            total_amount += float(txn["amount_inr"])

            diag = self.diagnostician.diagnose(txn)
            decision = self.strategist.decide(
                txn_id=txn["txn_id"],
                failure_reason=diag.predicted_reason,
                customer_segment=txn["customer_segment"],
            )

            fraud_score = float(txn.get("fraud_score", 0.0))
            review = self.guardrail.review(
                txn_id=txn["txn_id"],
                fraud_score=fraud_score,
                proposed_action=decision.chosen_action,
            )

            by_reason[diag.predicted_reason]["seen"] += 1

            if review.approved:
                # Only executed decisions feed the bandit -- a vetoed action
                # never ran, so it must never be treated as a real outcome.
                reward = self.strategist.simulate_reward(decision)
                self.strategist.update(decision, reward)
                recovered = reward == 1.0
                if recovered:
                    recovered_amount += float(txn["amount_inr"])
                    recovered_count += 1
                    by_reason[diag.predicted_reason]["recovered_amount"] += float(txn["amount_inr"])
                    by_reason[diag.predicted_reason]["recovered_count"] += 1
            else:
                recovered = False
                held_count += 1
                held_amount += float(txn["amount_inr"])

            msg = self.communicator.compose(
                txn_id=txn["txn_id"],
                action=review.final_action,
                customer_name=random.choice(FIRST_NAMES),
                amount=float(txn["amount_inr"]),
            )

            self.trace.append(
                {
                    "txn_id": txn["txn_id"],
                    "amount_inr": txn["amount_inr"],
                    "diagnosis": asdict(diag),
                    "decision": {
                        "chosen_action": decision.chosen_action,
                        "reasoning": decision.reasoning,
                        "explored": decision.explored,
                    },
                    "guardrail": {
                        "fraud_score": review.fraud_score,
                        "approved": review.approved,
                        "reasoning": review.reasoning,
                    },
                    "message": asdict(msg),
                    "recovered": recovered,
                    "held_for_review": not review.approved,
                }
            )

        summary = {
            "transactions_processed": len(df),
            "total_amount_at_risk_inr": round(total_amount, 2),
            "recovered_count": recovered_count,
            "recovered_amount_inr": round(recovered_amount, 2),
            "recovery_rate": round(recovered_count / len(df), 4) if len(df) else 0,
            "held_for_review_count": held_count,
            "held_for_review_amount_inr": round(held_amount, 2),
            "recovered_by_reason": {
                reason: {
                    "recovered_amount_inr": round(v["recovered_amount"], 2),
                    "recovered_count": v["recovered_count"],
                    "seen": v["seen"],
                }
                for reason, v in by_reason.items()
            },
        }
        self.metrics["batch_summary"] = summary
        return summary

    def guardrail_stats(self):
        """Cumulative Guardrail Agent activity across every batch run so far."""
        return {
            "reviews": self.guardrail.review_count,
            "vetoes": self.guardrail.veto_count,
            "veto_rate": round(self.guardrail.veto_count / self.guardrail.review_count, 4)
            if self.guardrail.review_count
            else 0,
        }

    def top_learned_policies(self, k: int = 8):
        """Returns the current best action per (reason, segment) context,
        i.e. what the Strategist has learned so far -- useful to show in
        the dashboard as proof the system is actually learning."""
        rows = []
        for context, actions in self.strategist.Q.items():
            best_action = max(actions, key=actions.get)
            rows.append(
                {
                    "failure_reason": context[0],
                    "customer_segment": context[1],
                    "best_action": best_action,
                    "estimated_recovery_prob": round(actions[best_action], 3),
                    "samples": self.strategist.counts[context][best_action],
                }
            )
        rows.sort(key=lambda r: -r["estimated_recovery_prob"])
        return rows[:k]
