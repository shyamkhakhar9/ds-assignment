# Telco Customer Churn Prediction

End-to-end machine learning project that predicts whether a telecom customer is likely to churn, then serves the trained pipeline through a REST API.

Business problem → data preparation → EDA → feature engineering → Decision Tree → evaluation → interpretation → saved pipeline → API.

## Dataset

IBM Telco Customer Churn. Target: `Churn` (`Yes` / `No`).

Place the CSV at `data/Telco-Customer-Churn.csv`. A copy is included in this repo. To refresh it:

```bash
curl -L -o data/Telco-Customer-Churn.csv \
  "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
```

## Project layout

```
ds-assignment/
├── data/Telco-Customer-Churn.csv
├── notebook/churn_analysis.ipynb
├── src/preprocessing.py      # shared cleaning + features (notebook and API)
├── src/train.py              # trains two trees, saves the selected pipeline
├── model/churn_pipeline.pkl
├── app.py
├── requirements.txt
├── sample_request.json
└── sample_response.json
```

Train/test split is **70:30** with `random_state=42` and stratification on churn. Encoders and the tree are fit on the training fold only and persisted as one `sklearn` pipeline so new API rows get the same transforms.

## Setup

```bash
cd ds-assignment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
PYTHONPATH=. python -m src.train
```

This trains two Decision Tree baselines, a GridSearch-tuned tree, logistic regression, and a random forest. It writes the best **Decision Tree** pipeline to `model/churn_pipeline.pkl` and comparison metrics to `model/metrics.json`.

Open `notebook/churn_analysis.ipynb` for the full analysis (EDA, feature rationale, metrics, feature importance, and tree visualization). From the repo root:

```bash
jupyter notebook notebook/churn_analysis.ipynb
```

## Run the API

```bash
PYTHONPATH=. uvicorn app:app --reload --port 8000
```

- `GET /health`
- `POST /predict`
- Interactive docs: http://127.0.0.1:8000/docs

### Sample request

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Example response (`sample_response.json`):

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8191
}
```

Invalid payloads (wrong enums, extra fields, missing keys) return HTTP 422 from FastAPI/Pydantic. A missing model file returns HTTP 503.

## Notes

- **Recall vs precision:** for proactive retention, the selected tree prefers catching churners (recall).
- **Class imbalance:** `class_weight="balanced"` on trees, logistic regression, and random forest.
- **Hyperparameter tuning:** `GridSearchCV` (5-fold, scoring = recall) over tree depth, leaf/split size, and class weight.
- **Extra models:** logistic regression and random forest on the same split (comparison only; API still serves a Decision Tree).
- Engineered features: `num_services`, `avg_monthly_spend`, `fiber_month_to_month`.
