"""Train Decision Trees (including GridSearch), compare extra models, persist the selected tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_model_pipeline, encode_target, load_raw_frame, split_xy

RANDOM_STATE = 42
TEST_SIZE = 0.30


def metrics_dict(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return out


def score_pipeline(pipe, X_test, y_test) -> dict:
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    result = metrics_dict(y_test, pred, proba)
    result["report"] = classification_report(
        y_test, pred, target_names=["No", "Yes"], zero_division=0
    )
    return result


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

    results = {}
    pipelines = {}

    configs = {
        "dt_shallow": (
            DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=RANDOM_STATE),
            False,
        ),
        "dt_balanced_deeper": (
            DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "logreg_balanced": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            True,
        ),
        "rf_balanced": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
    }

    for name, (clf, scale) in configs.items():
        pipe = build_model_pipeline(clf, scale_numeric=scale)
        pipe.fit(X_train, y_train)
        results[name] = score_pipeline(pipe, X_test, y_test)
        pipelines[name] = pipe

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        build_model_pipeline(
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            scale_numeric=False,
        ),
        param_grid={
            "model__max_depth": [4, 6, 8, 10],
            "model__min_samples_leaf": [5, 10, 20],
            "model__min_samples_split": [2, 10],
            "model__class_weight": [None, "balanced"],
        },
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    results["dt_gridsearch"] = score_pipeline(search.best_estimator_, X_test, y_test)
    results["dt_gridsearch"]["best_params"] = search.best_params_
    results["dt_gridsearch"]["cv_recall"] = float(search.best_score_)
    pipelines["dt_gridsearch"] = search.best_estimator_

    tree_names = ["dt_shallow", "dt_balanced_deeper", "dt_gridsearch"]
    best_tree = max(tree_names, key=lambda k: (results[k]["recall"], results[k]["f1"]))
    best_overall = max(results, key=lambda k: (results[k]["recall"], results[k]["f1"]))
    best_pipe = pipelines[best_tree]

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, out)

    payload = {
        "saved_model": best_tree,
        "best_overall_on_test": best_overall,
        "selection_note": (
            "API pickle is the best Decision Tree (required model family). "
            "Logistic regression and random forest are comparison-only."
        ),
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
    slim = {
        "saved_model": best_tree,
        "best_overall_on_test": best_overall,
        "results": payload["results"],
    }
    print(json.dumps(slim, indent=2))
    print(f"Saved pipeline to {out}")


if __name__ == "__main__":
    main()
