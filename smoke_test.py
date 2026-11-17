import pandas as pd
from sklearn.model_selection import train_test_split
from grader_contracts.ensembles import EnsembleInput
from ensemble_models import run_ensemble_experiment

df = pd.read_csv("german.csv", sep=";")
y = df["Creditability"].to_numpy()
X = df.drop(columns=["Creditability"])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

res = run_ensemble_experiment(EnsembleInput(
    train_features=Xtr, train_target=ytr,
    test_features=Xte,  test_target=yte,
    random_seed=42,
))
print("model_name :", res.model_name)
print("roc_auc    :", round(res.roc_auc, 4))
print("accuracy   :", round(res.accuracy, 4))
print("precision  :", round(res.precision, 4))
print("recall     :", round(res.recall, 4))
print("f1         :", round(res.f1, 4))