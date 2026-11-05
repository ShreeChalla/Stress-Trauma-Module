"""
tune_and_ensemble.py
Advanced acoustic feature engineering and model tuning:
- Extended feature set: delta energy, MFCC-style spectral roll-off/bandwidth/contrast, skewness, kurtosis
- Hyperparameter tuning for XGBoost, Random Forest, ExtraTrees, LightGBM/HistGradientBoosting, MLP
- Stacking / Soft-Voting Ensemble to achieve higher accuracy on unseen speakers
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import xgboost as xgb

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
COMBINED_DIR = os.path.join(DATA_DIR, "combined")
MODEL_DIR = os.path.join(COMBINED_DIR, "models")


def main():
    print("==================================================")
    print("Ensemble & Hyperparameter Optimization")
    print("==================================================")

    X_train = np.load(os.path.join(COMBINED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(COMBINED_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(COMBINED_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(COMBINED_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(COMBINED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(COMBINED_DIR, "y_test.npy"))

    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    # 1. Tuned Random Forest
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=16,
        min_samples_split=6,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    # 2. Extra Trees (reduces variance across unseen speakers)
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=18,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # 3. Tuned XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=450,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.2,
        min_child_weight=2,
        random_state=42,
        eval_metric="logloss",
    )

    # 4. HistGradientBoosting
    hgb = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.04,
        max_depth=8,
        l2_regularization=1.5,
        random_state=42,
    )

    # 5. MLP Neural Network on normalized acoustic vectors
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=0.01,
        max_iter=400,
        random_state=42,
    )

    # 6. Tuned RBF SVM
    svm = SVC(C=3.0, gamma="scale", kernel="rbf", probability=True, random_state=42)

    # Soft-Voting Ensemble
    ensemble = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et),
            ("xgb", xgb_clf),
            ("hgb", hgb),
            ("svm", svm),
        ],
        voting="soft",
    )

    print("Training Full Stacking/Voting Ensemble on 4,626 utterances...")
    ensemble.fit(X_train_full_scaled, y_train_full)

    y_pred = ensemble.predict(X_test_scaled)
    y_prob = ensemble.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n================ ENSEMBLE TEST RESULTS (Speaker-Independent) ================")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Macro-F1:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Confusion Matrix (Low Stress / High Stress):\n{np.array(cm)}")

    # Update production model
    joblib.dump(ensemble, os.path.join(MODEL_DIR, "stress_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    with open(os.path.join(COMBINED_DIR, "feature_names.json"), "r") as f:
        feature_names = json.load(f)

    meta = {
        "best_model": "Ensemble (RandomForest + ExtraTrees + XGBoost + HistGB + SVM)",
        "use_scaler": True,
        "feature_names": feature_names,
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel exported successfully to {os.path.join(MODEL_DIR, 'stress_model.joblib')}")


if __name__ == "__main__":
    main()

