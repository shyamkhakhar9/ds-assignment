"""Train two Decision Tree configurations and persist the selected pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_model_pipeline, encode_target, load_raw_frame, split_xy

RANDOM_STATE = 42
TEST_SIZE = 0.30


def metrics_dict(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Telco-Customer-Churn.csv")
    parser.add_argument("--model-out", default="model/churn_pipeline.pkl")
    parser.add_argument("--metrics-out", default="model/metrics.json")
    args = parser.parse_args()

    df = load_raw_frame(args.data)
    X, y = split_xy(df)
    y_enc = encode_target(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )

    configs = {
        "shallow_interpretable": DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
        "balanced_deeper": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    pipelines = {}
    for name, clf in configs.items():
        pipe = build_model_pipeline(clf)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = metrics_dict(y_test, pred)
        results[name]["report"] = classification_report(
            y_test, pred, target_names=["No", "Yes"], zero_division=0
        )
        pipelines[name] = pipe

    # Prefer the config with higher recall, then F1 — retention wants to catch churners.
    best_name = max(results, key=lambda k: (results[k]["recall"], results[k]["f1"]))
    best_pipe = pipelines[best_name]

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, out)

    payload = {
        "best_model": best_name,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "results": {
            name: {k: v for k, v in metrics.items() if k != "report"}
            for name, metrics in results.items()
        },
        "reports": {name: metrics["report"] for name, metrics in results.items()},
    }
    Path(args.metrics_out).write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("best_model", "results")}, indent=2))
    print(f"Saved pipeline to {out}")


if __name__ == "__main__":
    main()
