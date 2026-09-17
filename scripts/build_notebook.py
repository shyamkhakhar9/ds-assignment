#!/usr/bin/env python3
"""Generate notebook/churn_analysis.ipynb."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
}

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


md(
    """# Telco Customer Churn Prediction

A telecommunications company wants to find customers who are likely to leave so the retention team can contact them first.

This notebook covers the full workflow: data understanding, preprocessing without leakage, EDA, feature engineering, Decision Tree models, evaluation, interpretation, and saving a reusable pipeline for the REST API.

**Dataset:** IBM Telco Customer Churn  
**Target:** `Churn` (`Yes` / `No`)  
**Split:** 70% train / 30% test, `random_state=42`"""
)

md(
    """## 1. Data understanding and preparation

Load the CSV, inspect types and quality issues, then build a sklearn pipeline that can be applied to unseen customers (including API requests)."""
)

code(
    """from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path.cwd().resolve()
if ROOT.name == "notebook":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    ID_COL,
    NUMERIC_FEATURES,
    TARGET_COL,
    TelcoFeatureEngineer,
    build_model_pipeline,
    encode_target,
    load_raw_frame,
    split_xy,
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4.5)

DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "model" / "churn_pipeline.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.30

df = load_raw_frame(DATA_PATH)
df.head()"""
)

code(
    """print("Shape:", df.shape)
print("\\nDtypes:")
print(df.dtypes)
print("\\nMissing values:")
print(df.isna().sum())
print("\\nDuplicate rows:", int(df.duplicated().sum()))
print("Duplicate customer IDs:", int(df[ID_COL].duplicated().sum()) if ID_COL in df.columns else "n/a")
print("\\nNumeric describe:")
display(df.describe())
print("Categorical uniques:")
for col in df.select_dtypes(include="object").columns:
    print(f"  {col}: {df[col].nunique()} values")"""
)

md(
    """**Observations**

- About 7,043 customers and 21 columns. `customerID` is an identifier and must not be used as a predictor.
- `TotalCharges` is stored as text. Blank strings appear for brand-new customers (`tenure = 0`); they are not picked up by `isna()` until we coerce to numeric.
- No duplicate customer IDs.
- `SeniorCitizen` is already 0/1. Other demographics and services are categorical strings.
- `Churn` is imbalanced (roughly three stayers for every churner), which matters for metric choice and tree `class_weight`."""
)

code(
    """df["TotalCharges_numeric"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Non-numeric TotalCharges rows:", int(df["TotalCharges_numeric"].isna().sum()))
display(df.loc[df["TotalCharges_numeric"].isna(), ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])
print("Churn counts:")
print(df[TARGET_COL].value_counts())
print("Churn rate:", (df[TARGET_COL] == "Yes").mean().round(4))"""
)

md(
    """### Preprocessing decisions

| Decision | Why |
| --- | --- |
| Drop `customerID` | Identifier; using it would leak nothing useful and would not exist the same way for scoring |
| Coerce `TotalCharges` and fill blanks with 0 | New customers have no billed total yet |
| One-hot encode categoricals with `handle_unknown="ignore"` | Same mapping for API JSON; unknown levels do not crash scoring |
| Fit the pipeline on **train only** | Avoids leakage from test-set category frequencies |
| Stratified 70:30 split, `random_state=42` | Assignment split plus stable class mix |
| Engineer features inside the pipeline | API rows get the same derived columns |

The shared code lives in `src/preprocessing.py` so the notebook and FastAPI stay aligned."""
)

code(
    """X, y = split_xy(df.drop(columns=["TotalCharges_numeric"], errors="ignore"))
y_enc = encode_target(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print("Train churn rate:", y_train.mean().round(4), " Test churn rate:", y_test.mean().round(4))"""
)

md(
    """## 2. Exploratory data analysis

Charts below use the **training** customers only so exploration does not peek at the hold-out set. Each figure has a short business takeaway."""
)

code(
    """eda = X_train.copy()
eda["Churn"] = np.where(y_train == 1, "Yes", "No")

fig, ax = plt.subplots()
sns.countplot(data=eda, x="Churn", order=["No", "Yes"], ax=ax)
ax.set_title("Churn distribution (train)")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
plt.tight_layout()
plt.show()"""
)

md(
    """**Insight:** Churners are the minority. A model that always predicts “No” would look accurate but would miss the customers the retention team actually needs."""
)

code(
    """fig, ax = plt.subplots()
sns.countplot(data=eda, x="Contract", hue="Churn", ax=ax)
ax.set_title("Churn by contract type")
plt.tight_layout()
plt.show()
print(pd.crosstab(eda["Contract"], eda["Churn"], normalize="index").round(3))"""
)

md(
    """**Insight:** Month-to-month customers churn far more than one- or two-year contracts. Lock-in and switching cost are strong retention levers."""
)

code(
    """fig, ax = plt.subplots()
sns.histplot(data=eda, x="tenure", hue="Churn", bins=20, element="step", ax=ax)
ax.set_title("Tenure distribution by churn")
plt.tight_layout()
plt.show()"""
)

md(
    """**Insight:** Risk is highest in the first months. Onboarding quality and early-life offers matter more than late-tenure discounts."""
)

code(
    """fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.countplot(data=eda, x="InternetService", hue="Churn", ax=axes[0])
axes[0].set_title("Churn by internet service")
sns.countplot(data=eda, x="PaymentMethod", hue="Churn", ax=axes[1])
axes[1].set_title("Churn by payment method")
axes[1].tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.show()"""
)

md(
    """**Insight:** Fiber customers and electronic-check payers show elevated churn. Service quality (fiber) and billing friction (e-check) are actionable themes for retention."""
)

code(
    """fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(data=eda, x="Churn", y="MonthlyCharges", ax=axes[0])
axes[0].set_title("Monthly charges vs churn")
sns.scatterplot(data=eda.sample(800, random_state=RANDOM_STATE), x="tenure", y="MonthlyCharges", hue="Churn", alpha=0.5, ax=axes[1])
axes[1].set_title("Monthly charges vs tenure")
plt.tight_layout()
plt.show()"""
)

md(
    """**Insight:** Churners tend to pay higher monthly charges, especially at low tenure — high bill, low loyalty. That combination is a natural targeting rule for the call centre."""
)

md(
    """## 3. Feature engineering

At least two extra features are created **inside** `TelcoFeatureEngineer` so they are rebuilt for every new customer."""
)

code(
    """engineer = TelcoFeatureEngineer()
feat_preview = engineer.fit_transform(X_train.head(8))
feat_preview[["tenure", "MonthlyCharges", "TotalCharges", "InternetService", "Contract", "num_services", "avg_monthly_spend", "fiber_month_to_month"]]"""
)

md(
    """| Feature | How it is created | Why it may help |
| --- | --- | --- |
| `num_services` | Count of `Yes` values across phone/internet add-ons | Customers with more products are usually stickier (higher switching cost). |
| `avg_monthly_spend` | `TotalCharges / tenure` (or `MonthlyCharges` if tenure is 0) | Captures realized spend vs list price; promotions and discounts show up here. |
| `fiber_month_to_month` | 1 if fiber **and** month-to-month | EDA showed both factors raise churn; the interaction is a compact high-risk flag. |

These columns are passed through with the original numerics; categoricals are one-hot encoded. Nothing is fit on the test set."""
)

md(
    """## 4. Model development

Two Decision Tree configurations:

1. **Shallow / interpretable** — `max_depth=4`, `min_samples_leaf=20` (easy to explain to the business).
2. **Deeper / class-balanced** — `max_depth=8`, `min_samples_leaf=10`, `class_weight="balanced"` (pushes the tree to notice the minority churn class).

The same preprocessing pipeline wraps both models."""
)

code(
    """configs = {
    "shallow_interpretable": DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=20, random_state=RANDOM_STATE
    ),
    "balanced_deeper": DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE
    ),
}

fitted = {}
rows = []
for name, clf in configs.items():
    pipe = build_model_pipeline(clf)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    fitted[name] = pipe
    rows.append(
        {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        }
    )

compare = pd.DataFrame(rows).set_index("model")
display(compare.round(4))"""
)

md(
    """**Selection:** For retention outreach we care most about **recall** (finding customers who will churn), then F1. The deeper balanced tree is preferred if it lifts recall without collapsing precision into noise. The shallow tree remains a useful explanation baseline."""
)

code(
    """best_name = compare.sort_values(["recall", "f1"], ascending=False).index[0]
best_pipe = fitted[best_name]
print("Selected model:", best_name)
print(best_pipe.named_steps["model"])"""
)

md(
    """## 5. Model evaluation"""
)

code(
    """y_pred = best_pipe.predict(X_test)
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1       :", round(f1_score(y_test, y_pred, zero_division=0), 4))
print()
print(classification_report(y_test, y_pred, target_names=["No", "Yes"], zero_division=0))

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["No", "Yes"], ax=ax, cmap="Blues")
ax.set_title(f"Confusion matrix — {best_name}")
plt.tight_layout()
plt.show()"""
)

md(
    """**Business reading of the matrix**

- **False negative (actual Yes, predicted No):** a churner we did not flag. The company loses the remaining lifetime value. This is the costly miss for a retention programme.
- **False positive (actual No, predicted Yes):** a stable customer who gets an extra call or discount. That wastes campaign budget but is usually cheaper than an unrecoverable churn.

**Precision or recall?** Prioritize **recall**. The assignment is to *identify customers who may churn* so the team can engage them. Missing churners (low recall) defeats that goal. Precision still matters so agents are not flooded with false alarms; `class_weight="balanced"` is a simple way to trade some precision for recall without changing the API."""
)

md(
    """## 6. Model interpretation"""
)

code(
    """preprocess = best_pipe.named_steps["preprocess"]
model = best_pipe.named_steps["model"]
feature_names = preprocess.get_feature_names_out()
importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
top = importances.head(15)

fig, ax = plt.subplots(figsize=(8, 6))
top.sort_values().plot(kind="barh", ax=ax)
ax.set_title("Top feature importances")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.show()
display(top.to_frame("importance"))"""
)

code(
    """fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(
    model,
    feature_names=feature_names,
    class_names=["No", "Yes"],
    filled=True,
    max_depth=3,
    fontsize=8,
    ax=ax,
)
ax.set_title("Decision tree (first 3 levels)")
plt.tight_layout()
plt.show()"""
)

md(
    """**Key findings**

- Contract type, tenure, and fiber / month-to-month risk typically sit near the top of the tree — consistent with EDA.
- Higher monthly charges and electronic check also split high-risk leaves.
- `num_services` and `avg_monthly_spend` add product-depth and realized-bill signal on top of the raw catalogue fields.
- The shallow depth-4 tree is easier to print on a slide; the selected model uses extra depth plus class weight to recover more churners.

These rules are associative, not causal, but they line up with how telecom retention teams already think: early tenure, flexible contracts, expensive fiber plans."""
)

md(
    """## 7. Save the pipeline

The pickle stores **features + encoding + tree**, which is what `POST /predict` loads."""
)

code(
    """MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(best_pipe, MODEL_PATH)
print("Wrote", MODEL_PATH)

# Sanity check: score one raw customer dict the same way the API will
sample = X_test.iloc[[0]]
proba = best_pipe.predict_proba(sample)[0]
label = int(best_pipe.predict(sample)[0])
print("Sample prediction:", "Yes" if label == 1 else "No", "p(churn)=", round(float(proba[list(best_pipe.classes_).index(1)]), 4))
sample.iloc[0].to_dict()"""
)

md(
    """## How to run the API

```bash
PYTHONPATH=. uvicorn app:app --reload --port 8000
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```

See `README.md` for setup, `sample_request.json`, and `sample_response.json`."""
)

nb["cells"] = cells
out = Path(__file__).resolve().parents[1] / "notebook" / "churn_analysis.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("Wrote", out)
