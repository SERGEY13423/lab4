"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
)

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

FEATURE_COLUMNS = [
    "Account_Balance", "Duration_of_Credit_monthly",
    "Payment_Status_of_Previous_Credit", "Purpose", "Credit_Amount",
    "Value_Savings_Stocks", "Length_of_current_employment",
    "Instalment_per_cent", "Sex_Marital_Status", "Guarantors",
    "Duration_in_Current_address", "Most_valuable_available_asset",
    "Age_years", "Concurrent_Credits", "Type_of_apartment",
    "No_of_Credits_at_this_Bank", "Occupation", "No_of_dependents",
    "Telephone", "Foreign_Worker",
]

def _to_frame(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    df = pd.DataFrame(x)
    if df.shape[1] == len(FEATURE_COLUMNS):
        df.columns = FEATURE_COLUMNS
    return df

# Эту функцию мы возвращаем, чтобы experiments.py мог её использовать
def _add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering, одинаковый для train и test."""
    X = X.copy()
    X["Credit_per_month"] = X["Credit_Amount"] / X["Duration_of_Credit_monthly"]
    X["Age_at_start"]     = X["Age_years"] - X["Duration_of_Credit_monthly"] / 12.0
    X["log_Credit"]       = np.log1p(X["Credit_Amount"])
    X["log_Duration"]     = np.log1p(X["Duration_of_Credit_monthly"])
    X["Amount_x_Purpose"] = X["Credit_Amount"] * X["Purpose"]
    return X

def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучение Random Forest и возврат метрик."""
    seed = int(data.random_seed)

    # Используем сырые признаки, так как RF (raw) показал лучший результат
    X_train = _to_frame(data.train_features)
    X_test  = _to_frame(data.test_features)
    y_train = np.asarray(data.train_target).ravel()
    y_test  = np.asarray(data.test_target).ravel()

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    return EnsembleResult(
        model_name="RandomForest (raw)",
        probabilities=proba,
        predictions=pred,
        roc_auc=float(roc_auc_score(y_test, proba)),
        accuracy=float(accuracy_score(y_test, pred)),
        precision=float(precision_score(y_test, pred, zero_division=0)),
        recall=float(recall_score(y_test, pred, zero_division=0)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
    )