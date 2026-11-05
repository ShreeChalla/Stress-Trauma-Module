"""
train_models.py
Trains and evaluates Classic ML models for Binary Voice Stress Detection:
1. Random Forest Classifier
2. Gradient Boosting Classifier (XGBoost / HistGradientBoosting)
3. Support Vector Classifier (RBF kernel)

Features:
- Loads speaker-independent X_train, y_train, X_val, y_val, X_test, y_test
- Evaluates Accuracy, Macro-F1, Precision, Recall, ROC-AUC, Confusion Matrix
- Extracts top feature importances
- Saves optimal model as stress_model.joblib and scaler.joblib
- Generates a comprehensive training metrics report
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
COMBINED_DIR = os.path.join(DATA_DIR, "combined")
MODEL_DIR = os.path.join(COMBINED_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def evaluate_model(name: str, model, X_test, y_test, feature_names=None):
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n==================== {name} Results ====================")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Macro-F1:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Confusion Matrix (Low Stress / High Stress):\n{np.array(cm)}")

    # Feature Importance if available
    feat_imp = {}
    if hasattr(model, "feature_importances_") and feature_names:
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        print("\nTop Acoustic Feature Importances:")
        for idx in sorted_indices:
            print(f"  - {feature_names[idx]}: {importances[idx]:.4f}")
            feat_imp[feature_names[idx]] = float(importances[idx])

    return {
        "model": name,
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
        "feature_importances": feat_imp,
    }


def main():
    print("==================================================")
    print("Multi-Dataset Voice Stress Model Training & Evaluation")
    print("==================================================")

    # 1. Load data
    X_train = np.load(os.path.join(COMBINED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(COMBINED_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(COMBINED_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(COMBINED_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(COMBINED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(COMBINED_DIR, "y_test.npy"))

    with open(os.path.join(COMBINED_DIR, "feature_names.json"), "r") as f:
        feature_names = json.load(f)

    print(f"Training set:   {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set:       {X_test.shape[0]} samples")

    # Combine Train + Val for final model fitting
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    # 2. Fit Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_train_full_scaled = scaler.fit_transform(X_train_full)

    results = []

    # Model 1: Random Forest
    print("\nTraining Random Forest (300 estimators)...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_full, y_train_full)
    rf_metrics = evaluate_model("Random Forest", rf, X_test, y_test, feature_names)
    results.append(rf_metrics)

    # Model 2: XGBoost Classifier
    print("\nTraining XGBoost Classifier...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_clf.fit(X_train_full, y_train_full)
    xgb_metrics = evaluate_model("XGBoost", xgb_clf, X_test, y_test, feature_names)
    results.append(xgb_metrics)

    # Model 3: Support Vector Classifier (RBF kernel)
    print("\nTraining RBF SVM...")
    svm_clf = SVC(C=2.0, kernel="rbf", probability=True, random_state=42)
    svm_clf.fit(X_train_full_scaled, y_train_full)
    svm_metrics = evaluate_model("Support Vector Machine (RBF)", svm_clf, X_test_scaled, y_test, feature_names)
    results.append(svm_metrics)

    # Select Best Model based on Macro-F1
    best_res = max(results, key=lambda x: x["macro_f1"])
    print(f"\n================ WINNING MODEL: {best_res['model']} ================")
    print(f"Accuracy: {best_res['accuracy']*100:.2f}% | Macro-F1: {best_res['macro_f1']:.4f}")

    if best_res["model"] == "Random Forest":
        best_model = rf
        use_scaler = False
    elif best_res["model"] == "XGBoost":
        best_model = xgb_clf
        use_scaler = False
    else:
        best_model = svm_clf
        use_scaler = True

    # Persist model and scaler
    model_save_path = os.path.join(MODEL_DIR, "stress_model.joblib")
    scaler_save_path = os.path.join(MODEL_DIR, "scaler.joblib")
    meta_save_path = os.path.join(MODEL_DIR, "model_metadata.json")

    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    metadata = {
        "best_model": best_res["model"],
        "use_scaler": use_scaler,
        "feature_names": feature_names,
        "all_model_results": results,
    }
    with open(meta_save_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved best model to: {model_save_path}")
    print(f"Saved metadata to:   {meta_save_path}")


if __name__ == "__main__":
    main()

