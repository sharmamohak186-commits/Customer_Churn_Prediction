import pandas as pd
from pathlib import Path

def test_sample_dataset_has_expected_columns():
    path = Path(__file__).parents[1] / "data" / "churn_sample.csv"
    data = pd.read_csv(path)
    assert {"tenure_months", "monthly_charges", "support_tickets", "contract_type", "churn"} <= set(data.columns)
