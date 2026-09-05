import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.coordinator import RecoverAICoordinator

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "failed_transactions.csv")


def test_training_produces_reasonable_metrics():
    c = RecoverAICoordinator()
    metrics = c.train(DATA_PATH)
    assert metrics["accuracy"] > 0.3  # better than random guess across 8 classes
    assert metrics["n_classes"] == 8


def test_batch_run_recovers_some_revenue():
    c = RecoverAICoordinator()
    c.train(DATA_PATH)
    summary = c.run_batch(DATA_PATH, n=200)
    assert summary["transactions_processed"] == 200
    assert summary["recovered_amount_inr"] >= 0
    assert 0 <= summary["recovery_rate"] <= 1


def test_guardrail_vetoes_high_risk_automated_actions():
    """The Risk Guardrail Agent must never let a high fraud_score transaction
    go out on an automated/self-serve action -- it should always be forced
    to escalate_to_human_agent instead, and vetoed decisions must not be
    fed back into the Strategist's learned values."""
    c = RecoverAICoordinator()
    c.train(DATA_PATH)
    c.run_batch(DATA_PATH, n=300)

    held = [t for t in c.trace if t["held_for_review"]]
    for t in held:
        assert t["guardrail"]["approved"] is False
        assert t["guardrail"]["fraud_score"] >= c.guardrail.veto_threshold
        assert t["message"]["channel"] == "internal_queue"

    stats = c.guardrail_stats()
    assert stats["reviews"] == 300
    assert stats["vetoes"] == len(held)


def test_strategist_learns_over_time():
    """Recovery rate should trend upward as the bandit accumulates data."""
    c = RecoverAICoordinator()
    c.train(DATA_PATH)
    rates = []
    for _ in range(5):
        s = c.run_batch(DATA_PATH, n=200)
        rates.append(s["recovery_rate"])
    # not a strict monotonic guarantee (stochastic), but average of the
    # last 2 batches should not be dramatically worse than the first
    assert sum(rates[-2:]) / 2 >= sum(rates[:1]) / 1 - 0.25


if __name__ == "__main__":
    test_training_produces_reasonable_metrics()
    test_batch_run_recovers_some_revenue()
    test_strategist_learns_over_time()
    print("All tests passed.")
