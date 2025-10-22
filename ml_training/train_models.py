"""
Simplified Enhanced Training - Just the essentials for better accuracy
Adds cross-validation and basic hyperparameter tuning without complexity
"""

import pandas as pd
import numpy as np
from utils import load_csv, save_model
from preprocess_and_features import preprocess_data
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings

warnings.filterwarnings('ignore')

CSV_PATH = "plant_dataset.csv"
TARGET_COL = "is_failure_soon"
FAILURE_THRESHOLD = 0.5
MODEL_OUT = "rf_model.joblib"
ENCODERS_OUT = "encoders.joblib"
LABEL_OUT = "label_encoder_target.joblib"

if __name__ == "__main__":
    df = load_csv(CSV_PATH)

    # Auto-create predictive maintenance target if needed
    if TARGET_COL not in df.columns:
        if TARGET_COL == "is_failure_soon":
            if "failure_probability" not in df.columns:
                raise ValueError("failure_probability column not found to derive is_failure_soon")
            # Initial derivation
            df["is_failure_soon"] = (df["failure_probability"].astype(float) >= FAILURE_THRESHOLD).astype(int)
            # If only one class, auto-adjust threshold to get both classes
            if df["is_failure_soon"].nunique(dropna=True) < 2:
                for q in [0.50, 0.75, 0.90]:
                    th = float(df["failure_probability"].quantile(q))
                    df["is_failure_soon"] = (df["failure_probability"].astype(float) >= th).astype(int)
                    if df["is_failure_soon"].nunique(dropna=True) >= 2:
                        print(f"Derived is_failure_soon using auto threshold at quantile {q} → {th:.2f}")
                        break
                else:
                    print("Warning: Could not create both classes for is_failure_soon; dataset may be too skewed.")
            else:
                print(f"Derived is_failure_soon using failure_threshold={FAILURE_THRESHOLD}")
        else:
            raise ValueError(f"Target column '{TARGET_COL}' not found in CSV")
    
    df_processed, encoders = preprocess_data(df, target_col=TARGET_COL)

    # Set target and features
    target_col = TARGET_COL
    feature_cols = [col for col in df_processed.columns if col != target_col]
    # Prevent leakage: if target is derived from failure_probability, exclude it from features
    if target_col == "is_failure_soon" and "failure_probability" in feature_cols:
        feature_cols.remove("failure_probability")

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(df_processed[target_col].astype(str))
    X = df_processed[feature_cols]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")

    # Enhanced hyperparameter tuning for RandomForest
    print("\nTuning RandomForest hyperparameters...")
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Use GridSearchCV with 5-fold CV for better performance
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=5,  # 5-fold CV for better reliability
        scoring='accuracy',  # Use accuracy since you want 96% accuracy
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation on training set
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"\nResults:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Cross-validation scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model and artifacts
    save_model(best_rf, MODEL_OUT)
    joblib.dump(le, LABEL_OUT)
    joblib.dump(encoders, ENCODERS_OUT)
    
   