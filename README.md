# 💡 RecoverAI — Agentic Payment Recovery Copilot

**Razorpay AI Builder Internship 2026 — Track 3: AI Revenue Recovery**
*(architecture also extends into Track 2: AI Risk Manager, via the Risk Guardrail Agent below)*

> Every payment gateway loses a meaningful slice of GMV to **failed transactions** — timeouts, expired OTPs, insufficient funds, risk-engine false positives. Most of that money is *recoverable* if you react to the right failure with the right action, fast. RecoverAI is a self-improving, explainable multi-agent system that diagnoses *why* a payment failed, decides *how* to win it back, checks that decision against fraud risk, and *learns* which recovery strategy works best for which customer — autonomously, transaction by transaction.

Built solo by **[Sujal Singhal](https://github.com/SujalSinghal123)** for the Razorpay AI Builder Internship 2026 application.

---

## 🚨 The Problem

Razorpay processes payments for hundreds of thousands of merchants. Industry data puts **payment failure rates at 5–15%** depending on method and geography, and a large fraction of that revenue is simply written off because:

- Merchants treat every failure the same way (one generic "please retry" message).
- Nobody diagnoses the *root cause* of the failure at transaction time.
- Recovery strategy is static — it never learns from what actually worked.
- There's no explainability, so no one can audit *why* the system took a given action (a real requirement in fintech).
- A learning/exploring policy is also exactly the kind of thing you don't want making unsupervised contact with a customer on a transaction that looks fraudulent.

## 🧠 The Idea

RecoverAI is a **4-agent pipeline** that runs on every failed transaction:

```
Failed Transaction
       │
       ▼
┌──────────────────────┐   "Why did this fail?"
│  Diagnostician Agent  │   RandomForest classifier over txn features
│  (ML root-cause)      │   → predicted failure_reason + confidence
└─────────┬─────────────┘
          ▼
┌──────────────────────┐   "What should we do about it?"
│  Strategist Agent     │   Contextual multi-armed bandit
│  (adaptive policy)    │   (epsilon-greedy over reason × segment)
└─────────┬─────────────┘   → learns the best recovery action over time
          ▼
┌──────────────────────┐   "Is this actually safe to auto-send?"
│  Risk Guardrail Agent │   Vetoes automated actions above a fraud-score
│  (safety layer)       │   threshold, forcing human escalation instead
└─────────┬─────────────┘   → a vetoed action never reaches the bandit as a fake outcome
          ▼
┌──────────────────────┐   "Tell the customer, the right way."
│  Communicator Agent   │   Personalized message per channel
│  (channel + tone)     │   (WhatsApp / email / SMS / human escalation)
└─────────┬─────────────┘   Pluggable: template engine ↔ Claude API
          ▼
   Recovery outcome ──────► fed back into Strategist (closes the loop)
```

**Why this is different from a rules engine:** the Strategist Agent doesn't hard-code "if reason=X then action=Y". It maintains a live estimate of recovery probability for every `(failure_reason, customer_segment) → action` pair and updates it from real outcomes, balancing exploration vs. exploitation (epsilon-greedy bandit). The dashboard literally shows the policy improving as more transactions flow through — this is the "agentic" part: the system's behavior isn't static, it adapts.

**Why the Guardrail Agent matters:** an adaptive, exploring policy is a liability the moment it can autonomously message a high-risk transaction. The Guardrail scores every transaction's fraud risk and vetoes any self-serve action (retry, payment link, pay-later) above threshold, forcing escalation to a human agent instead — and critically, a vetoed decision is never fed back into the bandit as if it had actually run, so the learning loop can't be corrupted by outcomes that never happened.

**Why this matters for Razorpay specifically:** this isn't a generic ML toy — it's built around the actual shape of Razorpay's data (UPI/card/netbanking/wallet failure patterns, bank response codes, retry counts, a fraud/risk score) and directly targets a metric a merchant success team would care about: **₹ revenue recovered**, alongside a metric a risk team would care about: **% of automated actions correctly held for review**.

## 📊 Live Demo Results (synthetic data, 4,000 failed transactions)

| Metric | Value |
|---|---|
| Diagnostician accuracy (8-way failure classification) | ~52% (vs. 12.5% random baseline) |
| Simulated recovery rate | ~40–45% of failed transactions recovered |
| Simulated revenue recovered per 300-txn batch | ~₹1.2–2L |
| Guardrail veto rate | ~5–7% of proposed actions held for human review |
| Fully explainable | Every decision has a human-readable reasoning trace |

Run it yourself and these numbers will vary slightly (bandit exploration is stochastic by design) — that's expected and demonstrates the learning loop.

## 🖥️ Dashboard

A single-page, zero-build-step dashboard (Flask + vanilla HTML/CSS/JS) shows:
- The 4-agent pipeline, with a live pulse across the steps on every run.
- A running ledger of transactions processed, recovery rate, revenue recovered, and transactions held by the Guardrail.
- "What the Strategist has learned" — the current best action per (failure reason, segment), live.
- Recovered revenue by failure reason, and a recovery-rate trend across runs.
- A live reasoning trace: every agent's decision and *why*, per transaction, styled as a receipt/ledger feed — so the whole pipeline is auditable end to end.

## 🏗️ Repository Structure

```
recoverai/
├── agents/
│   ├── diagnostician.py    # ML root-cause classifier (RandomForest)
│   ├── strategist.py       # Contextual bandit recovery-action policy
│   ├── guardrail.py        # Risk Guardrail Agent (fraud-score veto layer)
│   ├── communicator.py     # Personalized message generation
│   └── coordinator.py      # Orchestrates the full pipeline + metrics
├── data/
│   └── generate_transactions.py   # Synthetic Razorpay-style failed-txn dataset
├── dashboard/
│   ├── app.py               # Flask backend
│   └── templates/index.html # Live single-page dashboard
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── LICENSE
└── README.md
```

## ⚙️ Setup

```bash
git clone <this-repo-url>
cd recoverai
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Generate synthetic data
python data/generate_transactions.py

# 2. Run tests
python tests/test_pipeline.py

# 3. Launch the dashboard
cd dashboard && python app.py
# open http://localhost:5000
```

Click **"Run new recovery batch"** on the dashboard to watch all four agents reason through a fresh batch of failed transactions in real time, and watch the "What the Strategist Has Learned" table update as the policy improves.

### Optional: real LLM-generated messages
Set `ANTHROPIC_API_KEY` in your environment before launching the dashboard, and the Communicator Agent will call the Claude API to generate natural-language recovery messages instead of using templates — with automatic fallback to templates if the call fails.

## 🧩 Build Challenges & How They Were Solved

1. **No labeled "which recovery action works" data exists (cold start problem).**
   Solved by modeling recovery as a **contextual bandit** instead of supervised learning — the system starts with optimistic initial estimates per context and *learns online* from outcomes, rather than needing a pre-labeled dataset of "successful recovery strategies" that doesn't exist yet in the real world either.

2. **Avoiding a black-box system in a fintech context.**
   Every agent emits a structured, human-readable `reasoning` string alongside its decision (confidence scores, top contributing features, exploration vs. exploitation flag, guardrail veto rationale). This is surfaced directly in the dashboard's live trace — critical for any system that will eventually touch real customer payments and needs to be auditable.

3. **Balancing exploration vs. exploitation without hurting real customers.**
   Used epsilon-greedy with a conservative epsilon (0.15) plus a domain-knowledge-constrained action space (`ELIGIBLE_ACTIONS`) — the bandit only explores among actions that are *plausible* for a given failure reason (e.g. it will never "instant retry" a risk-engine block), so exploration never proposes obviously wrong actions.

4. **An exploring policy is a liability on high-risk transactions.**
   Added a dedicated **Risk Guardrail Agent** as a hard veto layer between the Strategist and the Communicator: above a fraud-score threshold, only human escalation is allowed, no matter what the bandit proposes. Vetoed decisions are excluded from the bandit's reward updates so a blocked action can never masquerade as a real outcome.

5. **Class imbalance in failure reasons.**
   `RandomForestClassifier(class_weight="balanced")` plus macro-F1 as the evaluation metric (not just accuracy) so rare-but-important failure types (e.g. `risk_engine_block`) aren't ignored by the model.

6. **Making it runnable without any paid API keys.**
   The Communicator Agent defaults to a template engine and only calls the Claude API if a key is explicitly provided — so the entire system is fully demoable offline/free, while still showing a clear production upgrade path.

## 🚀 What's Next (Production Roadmap)

- Replace synthetic data with real Razorpay transaction/webhook streams.
- Swap the bandit for a contextual bandit with real feature embeddings (e.g. LinUCB) for faster convergence.
- Replace the Guardrail's threshold rule with a trained fraud model, and add a feedback loop so it also learns from confirmed fraud/chargeback outcomes.
- A/B test against Razorpay's current retry logic on live merchant traffic.
- Merchant-facing analytics: recovered revenue by merchant, by failure category, over time.

---

*Built solo by [Sujal Singhal](https://github.com/SujalSinghal123) for the Razorpay AI Builder Internship 2026 application.*

