"""
RecoverAI Dashboard
--------------------
Minimal Flask app that trains the pipeline once at startup, exposes
/api/run to simulate a new recovery batch on demand, and renders a
single-page dashboard (funnel metrics, learned policies, live agent
reasoning trace) -- built with zero frontend build step so it runs
anywhere with just `python app.py`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, render_template

from agents.coordinator import RecoverAICoordinator

app = Flask(__name__)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "failed_transactions.csv")

coordinator = RecoverAICoordinator()
train_metrics = coordinator.train(DATA_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/train_metrics")
def api_train_metrics():
    return jsonify(train_metrics)


@app.route("/api/run")
def api_run():
    summary = coordinator.run_batch(DATA_PATH, n=300)
    return jsonify(
        {
            "summary": summary,
            "trace_sample": coordinator.trace[-10:],
            "top_policies": coordinator.top_learned_policies(8),
            "guardrail_stats": coordinator.guardrail_stats(),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
