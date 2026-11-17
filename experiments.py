"""Исследовательский скрипт: сравнение ансамблей на german.csv.

Запуск:  python experiments.py
Выход:   experiments.csv  (таблица метрик)
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ensemble_models import _add_features

SEED = 42
CSV_PATH = "german.csv"


def evaluate(name: str, model, X_tr, X_te, y_tr, y_te) -> dict:
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    pr = (p >= 0.5).astype(int)
    return {
        "model": name,
        "roc_auc":   round(float(roc_auc_score(y_te, p)), 4),
        "accuracy":  round(float(accuracy_score(y_te, pr)), 4),
        "precision": round(float(precision_score(y_te, pr, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_te, pr, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_te, pr, zero_division=0)), 4),
    }


def main() -> None:
    df = pd.read_csv(CSV_PATH, sep=";")
    y = df["Creditability"].to_numpy()
    X = df.drop(columns=["Creditability"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    X_train_fe = _add_features(X_train)
    X_test_fe  = _add_features(X_test)

    results = []

    # 1. RandomForest
    results.append(evaluate(
        "RandomForest (raw)",
        RandomForestClassifier(n_estimators=500, max_depth=8,
                               min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        X_train, X_test, y_train, y_test,
    ))

    # 2. ExtraTrees
    results.append(evaluate(
        "ExtraTrees (raw)",
        ExtraTreesClassifier(n_estimators=500, max_depth=8,
                             min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        X_train, X_test, y_train, y_test,
    ))

    # 3. GradientBoosting
    results.append(evaluate(
        "GradientBoosting (raw)",
        GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                   max_depth=3, subsample=0.8,
                                   random_state=SEED),
        X_train, X_test, y_train, y_test,
    ))

    # 4. HistGB без FE
    results.append(evaluate(
        "HistGB (raw)",
        HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                       max_leaf_nodes=31, min_samples_leaf=20,
                                       l2_regularization=1.0,
                                       early_stopping=True,
                                       validation_fraction=0.15,
                                       n_iter_no_change=30,
                                       random_state=SEED),
        X_train, X_test, y_train, y_test,
    ))

    # 5. HistGB + FE
    results.append(evaluate(
        "HistGB + FE",
        HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                       max_leaf_nodes=31, min_samples_leaf=20,
                                       l2_regularization=1.0,
                                       early_stopping=True,
                                       validation_fraction=0.15,
                                       n_iter_no_change=30,
                                       random_state=SEED),
        X_train_fe, X_test_fe, y_train, y_test,
    ))

    # 6. Stacking + FE
    base = [
        ("rf",  RandomForestClassifier(n_estimators=400, max_depth=8,
                                       min_samples_leaf=3, random_state=SEED, n_jobs=-1)),
        ("gb",  GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                           max_depth=3, subsample=0.8,
                                           random_state=SEED)),
        ("hgb", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                               max_leaf_nodes=15, min_samples_leaf=20,
                                               l2_regularization=1.0,
                                               early_stopping=True,
                                               validation_fraction=0.15,
                                               n_iter_no_change=20,
                                               random_state=SEED)),
    ]
    stack = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=1000, C=1.0),
        cv=5, n_jobs=-1,
    )
    results.append(evaluate(
        "Stacking (RF+GB+HGB) + FE", stack,
        X_train_fe, X_test_fe, y_train, y_test,
    ))

    # 7. MLP + FE (со StandardScaler)
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                              alpha=1e-3, learning_rate_init=1e-3,
                              max_iter=800, early_stopping=True,
                              validation_fraction=0.15, n_iter_no_change=30,
                              random_state=SEED)),
    ])
    results.append(evaluate(
        "MLP + FE", mlp,
        X_train_fe, X_test_fe, y_train, y_test,
    ))

    res_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    res_df.to_csv("experiments.csv", index=False)

    print("\n=== Результаты экспериментов ===\n")
    print(res_df.to_string(index=False))

    best = res_df.iloc[0].to_dict()
    print("\nЛучшая модель:", best["model"], "| ROC-AUC =", best["roc_auc"])
    print("Числа сохранены в experiments.csv")


if __name__ == "__main__":
    main()