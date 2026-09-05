"""
Diagnostician Agent
--------------------
Role: "Why did this payment fail?"

Trains a RandomForest classifier on transaction features to predict the
failure_reason, and exposes a `.diagnose()` method that returns a ranked,
explainable diagnosis (top predicted reason + confidence + top contributing
features) for a single incoming failed transaction.

This is Agent #1 in the RecoverAI pipeline. Its output feeds the
Strategist Agent, which decides the recovery action.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


FEATURE_COLS = [
    "amount_inr",
    "retry_count_before",
    "past_success_rate",
    "network_latency_ms",
    "hour_of_day",
    "is_weekend",
]
CATEGORICAL_COLS = ["payment_method", "device_type", "customer_segment"]


@dataclass
class Diagnosis:
    txn_id: str
    predicted_reason: str
    confidence: float
    top_features: List[str]
    reasoning: str


class DiagnosticianAgent:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
        )
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.target_encoder = LabelEncoder()
        self.feature_names_ = []
        self.is_trained = False

    def _encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # handle unseen categories gracefully
                df[col] = df[col].astype(str).map(
                    lambda v: v if v in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col])
        return df[FEATURE_COLS + CATEGORICAL_COLS]

    def train(self, csv_path: str) -> Dict[str, float]:
        df = pd.read_csv(csv_path)
        df["is_weekend"] = df["is_weekend"].astype(int)

        X = self._encode(df, fit=True)
        self.feature_names_ = list(X.columns)
        y = self.target_encoder.fit_transform(df["failure_reason"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)

        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1_macro": round(f1_score(y_test, preds, average="macro"), 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_classes": len(self.target_encoder.classes_),
        }
        self.is_trained = True
        return metrics

    def diagnose(self, txn: Dict) -> Diagnosis:
        if not self.is_trained:
            raise RuntimeError("DiagnosticianAgent must be trained before use.")

        row = pd.DataFrame([txn])
        row["is_weekend"] = row["is_weekend"].astype(int)
        X = self._encode(row, fit=False)

        probs = self.model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        reason = self.target_encoder.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])

        importances = self.model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:3]
        top_features = [self.feature_names_[i] for i in top_idx]

        reasoning = (
            f"Diagnosed as '{reason}' with {confidence:.0%} confidence. "
            f"Most influential signals (model-wide): {', '.join(top_features)}."
        )

        return Diagnosis(
            txn_id=txn.get("txn_id", "unknown"),
            predicted_reason=reason,
            confidence=round(confidence, 4),
            top_features=top_features,
            reasoning=reasoning,
        )
