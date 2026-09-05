"""
generate_transactions.py
-------------------------
Generates a realistic synthetic dataset of FAILED payment transactions for a
payment gateway (Razorpay-style). This stands in for real production data
(txn logs, bank response codes, customer history) that RecoverAI would
normally consume from Razorpay's transaction pipeline / webhooks.

Each row = one failed transaction with enough signal for the Diagnostician
Agent to classify WHY it failed, and for the Strategist Agent to decide
HOW to recover it.
"""

import random
import csv
import os
from datetime import datetime, timedelta

random.seed(42)

FAILURE_REASONS = [
    "insufficient_funds",
    "bank_server_timeout",
    "otp_expired",
    "card_expired",
    "risk_engine_block",
    "network_drop",
    "wrong_cvv",
    "daily_limit_exceeded",
]

PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet"]
DEVICE_TYPES = ["android", "ios", "web"]
CUSTOMER_SEGMENTS = ["new", "returning", "high_value", "at_risk_churn"]
BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "Yes Bank", "IDFC"]


def _pick_reason(method):
    # Certain failure reasons correlate with certain payment methods,
    # which is exactly the kind of pattern the ML classifier should learn.
    if method == "upi":
        return random.choices(
            ["otp_expired", "network_drop", "bank_server_timeout", "risk_engine_block"],
            weights=[35, 30, 20, 15],
        )[0]
    if method == "card":
        return random.choices(
            ["insufficient_funds", "card_expired", "wrong_cvv", "risk_engine_block", "daily_limit_exceeded"],
            weights=[30, 20, 20, 15, 15],
        )[0]
    if method == "netbanking":
        return random.choices(
            ["bank_server_timeout", "otp_expired", "network_drop"],
            weights=[45, 30, 25],
        )[0]
    return random.choices(
        ["insufficient_funds", "risk_engine_block", "network_drop"],
        weights=[40, 30, 30],
    )[0]


def generate(n=4000, out_path="failed_transactions.csv"):
    start = datetime(2026, 6, 1)
    rows = []
    for i in range(n):
        method = random.choice(PAYMENT_METHODS)
        reason = _pick_reason(method)
        segment = random.choices(
            CUSTOMER_SEGMENTS, weights=[30, 40, 20, 10]
        )[0]
        amount = round(random.lognormvariate(6.5, 1.1), 2)  # skewed, realistic txn amounts
        hour = random.randint(0, 23)
        ts = start + timedelta(
            days=random.randint(0, 89), hours=hour, minutes=random.randint(0, 59)
        )
        retry_count_before = random.choices([0, 1, 2, 3], weights=[70, 18, 8, 4])[0]
        past_success_rate = round(random.betavariate(5, 2) if segment != "new" else random.betavariate(1, 3), 3)
        network_latency_ms = random.randint(80, 4000) if reason in ("bank_server_timeout", "network_drop") else random.randint(80, 600)

        # Fraud/risk score in [0, 1] -- feeds the Risk Guardrail Agent.
        # Correlated with real risk signals: risk_engine_block reason, very
        # high amounts, brand-new customers, and repeated retries all push
        # the score up, so the guardrail has genuine signal to act on.
        risk = 0.05
        if reason == "risk_engine_block":
            risk += 0.55
        if amount > 15000:
            risk += 0.2
        if segment == "new":
            risk += 0.1
        if retry_count_before >= 2:
            risk += 0.1
        risk += random.uniform(-0.05, 0.05)
        fraud_score = round(min(max(risk, 0.01), 0.99), 3)

        rows.append(
            {
                "txn_id": f"pay_{100000+i}",
                "timestamp": ts.isoformat(),
                "amount_inr": amount,
                "payment_method": method,
                "bank": random.choice(BANKS) if method in ("card", "netbanking") else "",
                "device_type": random.choice(DEVICE_TYPES),
                "customer_segment": segment,
                "failure_reason": reason,
                "retry_count_before": retry_count_before,
                "past_success_rate": past_success_rate,
                "network_latency_ms": network_latency_ms,
                "hour_of_day": hour,
                "is_weekend": ts.weekday() >= 5,
                "fraud_score": fraud_score,
            }
        )

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {n} synthetic failed transactions -> {out_path}")
    return out_path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    generate(4000, os.path.join(here, "failed_transactions.csv"))
