# Customer Churn Prediction

A beginner-friendly machine learning project that predicts whether a customer is likely to leave a subscription service. It trains a scikit-learn classification model on a small illustrative dataset and exposes predictions through FastAPI.

## Skills demonstrated
Python, Pandas, data preprocessing, classification, train/test split, model evaluation, joblib model persistence, FastAPI, REST APIs.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs

Training creates `models/churn_model.joblib`. The included CSV is a small synthetic teaching dataset, not real customer data. For a serious portfolio version, replace it with a properly licensed public dataset and document its source.

## API
`POST /predict`
```json
{
  "tenure_months": 8,
  "monthly_charges": 75.5,
  "support_tickets": 4,
  "contract_type": "month-to-month"
}
```

## Evaluation
The training script prints accuracy, precision, recall, F1-score, and a confusion matrix. Do not report a metric in your resume until you have run the model and verified it on an appropriate dataset.
