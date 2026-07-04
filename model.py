"""
ML pipeline: trains a Random Forest crop-recommendation classifier on the
Indian Crop Recommendation dataset (N, P, K, temperature, humidity, pH,
rainfall -> crop). This mirrors the original deck's stated approach
("Random Forest Classifier trained on historical soil & climate datasets").
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_data(path="data/crop_recommendation.csv"):
    return pd.read_csv(path)


def train_model(df, random_state=42):
    X = df[FEATURES]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    return {
        "model": clf,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "labels": clf.classes_,
        "feature_importances": dict(zip(FEATURES, clf.feature_importances_)),
    }


def crop_nutrient_stats(df):
    """Per-crop mean/std for N, P, K -> used for data-driven fertilizer advisory."""
    stats = {}
    grouped = df.groupby("label")[["N", "P", "K"]].agg(["mean", "std"])
    for crop in grouped.index:
        stats[crop] = {
            "N": (grouped.loc[crop, ("N", "mean")], grouped.loc[crop, ("N", "std")]),
            "P": (grouped.loc[crop, ("P", "mean")], grouped.loc[crop, ("P", "std")]),
            "K": (grouped.loc[crop, ("K", "mean")], grouped.loc[crop, ("K", "std")]),
        }
    return stats


def predict_top_n(model, input_row, n=3):
    """input_row: dict with FEATURES keys. Returns list of (crop, probability)."""
    X = pd.DataFrame([input_row])[FEATURES]
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
    return ranked[:n]
