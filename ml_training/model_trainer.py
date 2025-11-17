#!/usr/bin/env python3
"""
model_trainer.py
Train a small set of classifiers on the processed predictive-maintenance dataset and choose the best performer.

Input: CSV from `DataPreprocessor` with a binary `target` column.
- Converts feature columns to numeric, filling missing values with zero.
- Splits data with stratified train/validation sets.
- Trains three models with compact hyperparameter grids:
    * LogisticRegression
    * RandomForestClassifier
    * ExtraTreesClassifier
- Scores models with 5-fold stratified ROC AUC and evaluates on a hold-out validation set.
- Saves:
    models/
      ├─ model.joblib
      ├─ model_leaderboard.csv
      ├─ per_model_best_params.json
      ├─ eval_report.json
      └─ feature_names.json

Usage:
    python model_trainer.py --data path/to/processed_dataset.csv --out_dir ml_training/models
Optional:
    --target-col TARGET_NAME   (default: target)
    --seed 42
"""

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_matrix(path: str, target_col: str):
    df = pd.read_csv(path)
    # Try to locate target column; fallback to last column
    if target_col not in df.columns:
        target_col = df.columns[-1]

    y = df[target_col].astype(int).values
    X_df = df.drop(columns=[target_col]).copy()

    # Ensure numeric matrix (coerce any accidental strings)
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    X_df = X_df.fillna(0.0)

    feature_names = X_df.columns.tolist()
    X = X_df.values
    return X, y, feature_names


def main():
    """Train multiple models and select the best performer."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to processed CSV with `target` column")
    ap.add_argument("--out_dir", required=True, help="Directory to save models and results")
    ap.add_argument("--target-col", default="target", help="Name of target column (default: target)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    print("=" * 60)
    print("MODEL TRAINING - Predictive Maintenance")
    print("=" * 60)

    os.makedirs(args.out_dir, exist_ok=True)

    X, y, feature_names = load_matrix(args.data, args.target_col)
    uniq, cnt = np.unique(y, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(uniq, cnt)}

    print(f"Dataset rows: {len(X)}  |  features: {len(feature_names)}")
    print(f"Target distribution: {class_counts}")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25, random_state=args.seed, stratify=y
        )
    print(f"Train rows: {len(X_train)}  |  validation rows: {len(X_valid)}")

    # Three models with compact grids
    logreg_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=2000, random_state=args.seed))
    ])

    models = {
        "logreg": (
            logreg_pipeline,
            {
                "classifier__C": [0.5, 1.0, 5.0],
                "classifier__solver": ["liblinear", "lbfgs"],
            },
        ),
        "rf": (
            RandomForestClassifier(random_state=args.seed, n_jobs=-1),
            {
                "n_estimators": [200, 300],
                "max_depth": [5, 8],
                "min_samples_leaf": [2, 4],
                "min_samples_split": [5, 10],
                "max_features": ["sqrt"],
            },
        ),
        "extratrees": (
            ExtraTreesClassifier(random_state=args.seed, n_jobs=-1),
            {
                "n_estimators": [200, 300],
                "max_depth": [5, 8],
                "min_samples_leaf": [2, 4],
                "min_samples_split": [5, 10],
                "max_features": ["sqrt"],
            },
        ),
    }

    print(f"\nTraining {len(models)} models (5-fold CV)...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    leaderboard = []
    best_name, best_est, best_auc = None, None, -1.0
    best_params_all = {}

    model_names = {
        "logreg": "Logistic Regression",
        "rf": "Random Forest",
        "extratrees": "Extra Trees",
    }

    for idx, (name, (est, grid)) in enumerate(models.items(), 1):
        print(f"\nModel {idx}: {model_names.get(name, name)}")
        gs = GridSearchCV(
            estimator=est,
            param_grid=grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        best_params_all[name] = gs.best_params_
        print(f"  CV ROC AUC: {gs.best_score_:.4f}")

        # Get predictions for both train and validation to check overfitting
        estimator = gs.best_estimator_
        if hasattr(estimator, "predict_proba"):
            y_prob_train = estimator.predict_proba(X_train)[:, 1]
            y_prob = estimator.predict_proba(X_valid)[:, 1]
        else:
            y_score_train = estimator.decision_function(X_train)
            y_score = estimator.decision_function(X_valid)
            y_prob_train = (y_score_train - y_score_train.min()) / (y_score_train.max() - y_score_train.min() + 1e-9)
            y_prob = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-9)

        y_pred_train = (y_prob_train >= 0.5).astype(int)
        y_pred = (y_prob >= 0.5).astype(int)

        # Training metrics (for overfitting detection)
        train_accuracy = float(accuracy_score(y_train, y_pred_train))
        train_roc_auc = float(roc_auc_score(y_train, y_prob_train))

        # Validation metrics
        accuracy = float(accuracy_score(y_valid, y_pred))
        precision = float(precision_score(y_valid, y_pred, zero_division=0))
        recall = float(recall_score(y_valid, y_pred, zero_division=0))
        f1 = float(f1_score(y_valid, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_valid, y_prob))

        accuracy_gap = train_accuracy - accuracy
        roc_auc_gap = train_roc_auc - roc_auc
        if accuracy_gap > 0.15 or roc_auc_gap > 0.15:
            print(f"  Warning: large train/validation gap ({accuracy_gap:.2%})")

        metrics = {
            "model": name,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_roc_auc": roc_auc,
            "train_accuracy": train_accuracy,
            "train_roc_auc": train_roc_auc,
            "overfitting_gap": accuracy_gap,
        }
        leaderboard.append(metrics)

        print(f"  Validation: Accuracy {accuracy:.4f}, ROC AUC {roc_auc:.4f}, F1 {f1:.4f}")

        if metrics["val_roc_auc"] > best_auc:
            best_auc = metrics["val_roc_auc"]
            best_name = name
            best_est = gs.best_estimator_
            print("  -> current best model")

    if not leaderboard:
        raise RuntimeError("No models were successfully trained")

    lb_df = pd.DataFrame(leaderboard).sort_values("val_roc_auc", ascending=False)
    lb_path = os.path.join(args.out_dir, "model_leaderboard.csv")
    lb_df.to_csv(lb_path, index=False)

    print("\nValidation summary:")
    print(lb_df[["model", "val_accuracy", "val_roc_auc", "val_f1"]])

    if best_est is None:
        raise RuntimeError("Best model not found after training")

    model_path = os.path.join(args.out_dir, "model.joblib")
    joblib.dump(best_est, model_path)

    with open(os.path.join(args.out_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    # Optimize threshold for the selected model
    print("\nThreshold tuning...")
    if hasattr(best_est, "predict_proba"):
        y_prob = best_est.predict_proba(X_valid)[:, 1]
    else:
        y_score = best_est.decision_function(X_valid)
        y_prob = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-9)

    thresholds = np.arange(0.3, 0.8, 0.01)
    base_accuracy = accuracy_score(y_valid, (y_prob >= 0.5).astype(int))
    best_threshold = 0.5
    best_thresh_accuracy = base_accuracy

    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        acc = accuracy_score(y_valid, y_pred_thresh)
        if acc > best_thresh_accuracy:
            best_thresh_accuracy = acc
            best_threshold = thresh

    print(f"Best threshold: {best_threshold:.2f} (accuracy: {best_thresh_accuracy:.4f})")

    y_pred = (y_prob >= best_threshold).astype(int)

    final_accuracy = float(accuracy_score(y_valid, y_pred))
    final_precision = float(precision_score(y_valid, y_pred, zero_division=0))
    final_recall = float(recall_score(y_valid, y_pred, zero_division=0))
    final_f1 = float(f1_score(y_valid, y_pred, zero_division=0))
    final_roc_auc = float(roc_auc_score(y_valid, y_prob))

    report = {
        "target_distribution": class_counts,
        "best_model": best_name,
        "best_model_params": best_params_all.get(best_name, {}),
        "best_threshold": float(best_threshold),
        "val_metrics": {
            "accuracy": final_accuracy,
            "precision": final_precision,
            "recall": final_recall,
            "f1": final_f1,
            "roc_auc": final_roc_auc,
        },
        "files": {
            "leaderboard_csv": lb_path,
            "model_path": model_path,
        },
    }
    with open(os.path.join(args.out_dir, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Feature importance for tree-based models
    if best_name in ["rf", "extratrees"] and hasattr(best_est, "feature_importances_"):
        importances = best_est.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        print("\nTop 10 features:")
        for rank, idx in enumerate(indices, 1):
            print(f"  {rank:2d}. {feature_names[idx]:<30} {importances[idx]:.4f}")
        importance_dict = {
            feature_names[i]: float(importances[i]) for i in range(len(feature_names))
        }
        importance_path = os.path.join(args.out_dir, "feature_importance.json")
        with open(importance_path, "w") as f:
            json.dump(importance_dict, f, indent=2)

    print("\nSummary")
    print(f"  Best model: {model_names.get(best_name, best_name)}")
    print(f"  Threshold: {best_threshold:.2f}")
    print(f"  Accuracy:  {final_accuracy:.4f} | ROC AUC: {final_roc_auc:.4f} | F1: {final_f1:.4f}")
    print(f"Artifacts saved to {args.out_dir}")

    
if __name__ == "__main__":
    main()
