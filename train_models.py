# train_models.py

import argparse
import pandas as pd
from utils import load_csv, save_model
from preprocess_and_features import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier on plant dataset")
    parser.add_argument("--csv", default="plant_dataset.csv", help="Path to input CSV")
    parser.add_argument("--target", default="is_failure_soon", help="Target column to predict (default: is_failure_soon)")
    parser.add_argument("--failure_threshold", type=float, default=0.5, help="Threshold on failure_probability to create is_failure_soon if missing")
    parser.add_argument("--model_out", default="rf_model.joblib", help="Path to save trained model")
    parser.add_argument("--encoders_out", default="encoders.joblib", help="Path to save preprocessors")
    parser.add_argument("--label_out", default="label_encoder_target.joblib", help="Path to save target LabelEncoder")
    args = parser.parse_args()

    # Load and preprocess data
    df = load_csv(args.csv)

    # Auto-create predictive maintenance target if needed
    if args.target not in df.columns:
        if args.target == "is_failure_soon":
            if "failure_probability" not in df.columns:
                raise ValueError("failure_probability column not found to derive is_failure_soon")
            # Initial derivation
            df["is_failure_soon"] = (df["failure_probability"].astype(float) >= args.failure_threshold).astype(int)
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
                print(f"Derived is_failure_soon using failure_threshold={args.failure_threshold}")
        else:
            raise ValueError(f"Target column '{args.target}' not found in CSV")
    df_processed, encoders = preprocess_data(df, target_col=args.target)

    # Set target and features
    target_col = args.target
    feature_cols = [col for col in df_processed.columns if col != target_col]
    # Prevent leakage: if target is derived from failure_probability, exclude it from features
    if target_col == "is_failure_soon" and "failure_probability" in feature_cols:
        feature_cols.remove("failure_probability")

    # Encode target labels (keeps support for string targets)
    le = LabelEncoder()
    y = le.fit_transform(df_processed[target_col].astype(str))
    X = df_processed[feature_cols]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train a simple RandomForest (no class weighting, no CV, no threshold tuning)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    # Show results
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print("RandomForest Classification Report:")
    print(classification_report(y_test, preds, target_names=le.classes_))

    # Save model and artifacts
    save_model(rf, args.model_out)
    joblib.dump(le, args.label_out)
    joblib.dump(encoders, args.encoders_out)
