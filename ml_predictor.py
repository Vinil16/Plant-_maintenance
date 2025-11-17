from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd

from ml_training.data_preprocessing import DataPreprocessor


class MaintenancePredictor:
    """Load the persisted pipeline and make failure predictions for assets."""

    def __init__(self, models_dir: str | Path = "ml_training/models", data_path: str | Path = "plant_dataset.csv") -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline = None
        self.metadata: Dict[str, Any] = {}
        self.preprocessor = DataPreprocessor(data_path=data_path)
        self.prediction_threshold = 0.5
        self.expected_feature_names = None
        self.feature_importance = None
        self.training_stats = None

        self._load_metadata()
        self._load_pipeline()
        self._load_feature_importance()

    # ------------------------------------------------------------------
    def _load_metadata(self) -> None:
        # Try to load from artifacts.json (from preprocessing)
        artifacts_path = self.models_dir.parent / "artifacts.json"
        if artifacts_path.exists():
            with artifacts_path.open("r", encoding="utf-8") as meta_fp:
                self.metadata = json.load(meta_fp)
        else:
            self.metadata = {}

        # Load exact feature names from training
        feature_names_path = self.models_dir / "feature_names.json"
        if feature_names_path.exists():
            with feature_names_path.open("r", encoding="utf-8") as f:
                self.expected_feature_names = json.load(f)
        else:
            self.expected_feature_names = None

        numeric = self.metadata.get("numeric_features")
        categorical = self.metadata.get("categorical_features")
        if numeric:
            self.preprocessor.numeric_features = list(numeric)
        if categorical:
            self.preprocessor.categorical_features = list(categorical)

    def _load_pipeline(self) -> None:
        model_path = self.models_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. Run training before predicting."
            )
        self.pipeline = joblib.load(model_path)
        
        # Load threshold from eval_report if available
        eval_report_path = self.models_dir / "eval_report.json"
        if eval_report_path.exists():
            with eval_report_path.open("r", encoding="utf-8") as f:
                report = json.load(f)
                self.prediction_threshold = report.get("best_threshold", 0.5)
    
    def _load_feature_importance(self) -> None:
        """Load feature importance from saved JSON file."""
        importance_path = self.models_dir / "feature_importance.json"
        if importance_path.exists():
            with importance_path.open("r", encoding="utf-8") as f:
                self.feature_importance = json.load(f)
        
        # Load training dataset statistics for comparison
        processed_data_path = self.models_dir.parent / "processed_dataset.csv"
        if processed_data_path.exists():
            try:
                train_df = pd.read_csv(processed_data_path)
                if "target" in train_df.columns:
                    # Calculate statistics for numeric features
                    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
                    if "target" in numeric_cols:
                        numeric_cols.remove("target")
                    
                    self.training_stats = {}
                    for col in numeric_cols:
                        if col in train_df.columns:
                            self.training_stats[col] = {
                                "mean": float(train_df[col].mean()),
                                "median": float(train_df[col].median()),
                                "std": float(train_df[col].std()),
                                "q75": float(train_df[col].quantile(0.75)),
                                "q25": float(train_df[col].quantile(0.25)),
                            }
            except Exception:
                self.training_stats = None

    # ------------------------------------------------------------------
    def predict_failure(self, asset_id: str | None = None, df: pd.DataFrame | None = None) -> Dict[str, Any]:
        """Predict failure for an asset.
        
        Args:
            asset_id: Asset ID to predict for. If None and df is provided, uses first row.
            df: DataFrame containing asset data. If None, asset_id must be provided.
        
        Returns:
            Dictionary with prediction results or error information.
        """
        if self.pipeline is None:
            return {"error": "Pipeline not loaded. Train the model first."}

        try:
            # Handle different input formats
            if df is not None:
                if asset_id is not None:
                    # Find row by asset_id
                    asset_row = df[df.get("asset_id", pd.Series()) == asset_id]
                    if asset_row.empty:
                        return {"error": f"Asset {asset_id} not found in dataset."}
                    asset_dict = asset_row.iloc[0].to_dict()
                else:
                    # Use first row if no asset_id specified
                    asset_dict = df.iloc[0].to_dict()
            elif asset_id is not None:
                return {"error": "DataFrame required when providing asset_id."}
            else:
                return {"error": "Either asset_id with df, or asset dict required."}

            asset_df = pd.DataFrame([asset_dict])
            features = self._prepare_features(asset_df)
            probability = float(self.pipeline.predict_proba(features)[0, 1])
            predicted_class = int(probability >= self.prediction_threshold)
            
            # Get explanation for why it will fail
            explanation = self._explain_prediction(asset_dict, features[0], probability)

            return {
                "predicted_failure": bool(predicted_class),
                "ml_predicted_probability": probability,
                "failure_probability": probability,  # alias for compatibility
                "risk_level": self._risk_level(probability),
                "asset_id": asset_id or asset_dict.get("asset_id", "unknown"),
                "explanation": explanation,
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def assess_risk_level(self, asset_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess risk level for an asset."""
        result = self.predict_failure(asset_id=asset_id, df=df)
        if "error" in result:
            return {
                "success": False,
                "error": result["error"]
            }
        
        # Get asset details
        asset_row = df[df.get("asset_id", pd.Series()) == asset_id]
        asset_type = asset_row.iloc[0].get("asset_type", "Unknown") if not asset_row.empty else "Unknown"
        
        probability = result.get("failure_probability", 0.0)
        risk_level = result.get("risk_level", "low")
        
        # Calculate maintenance days based on probability
        maintenance_days = self._calculate_maintenance_days(probability)
        
        return {
            "success": True,
            "asset_id": asset_id,
            "asset_type": asset_type,
            "overall_risk": risk_level.title(),
            "failure_probability": probability,
            "risk_level": risk_level,
            "maintenance_days": maintenance_days,
            "predicted_failure": result.get("predicted_failure", False),
            "explanation": result.get("explanation", {})
        }

    def get_maintenance_schedule(self, asset_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Get maintenance schedule for an asset."""
        result = self.predict_failure(asset_id=asset_id, df=df)
        if "error" in result:
            return {
                "success": False,
                "error": result["error"]
            }
        
        # Get asset details
        asset_row = df[df.get("asset_id", pd.Series()) == asset_id]
        asset_type = asset_row.iloc[0].get("asset_type", "Unknown") if not asset_row.empty else "Unknown"
        
        probability = result.get("failure_probability", 0.0)
        risk_level = result.get("risk_level", "low")
        
        # Calculate maintenance days based on probability
        maintenance_days = self._calculate_maintenance_days(probability)
        
        return {
            "success": True,
            "asset_id": asset_id,
            "asset_type": asset_type,
            "recommended_days": maintenance_days,
            "maintenance_days": maintenance_days,
            "failure_probability": probability,
            "risk_level": risk_level,
            "predicted_failure": result.get("predicted_failure", False),
            "explanation": result.get("explanation", {})
        }

    def batch_predict(self, assets: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        rows = list(assets)
        if not rows:
            return pd.DataFrame()

        assets_df = pd.DataFrame(rows)
        features = self._prepare_features(assets_df)
        probabilities = self.pipeline.predict_proba(features)[:, 1]
        predictions = (probabilities >= self.prediction_threshold).astype(int)

        output = assets_df.copy()
        output["failure_probability"] = probabilities
        output["predicted_failure"] = predictions
        output["risk_level"] = [self._risk_level(p) for p in probabilities]
        return output

    # ------------------------------------------------------------------
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure preprocessor has been fitted by loading raw training data once
        if self.preprocessor._raw is None:
            # Load the full training dataset to fit the preprocessor
            self.preprocessor.load()
            # Run prepare to fit encoders and scalers (but don't save)
            bundle = self.preprocessor.prepare(save_processed=False)
            self.preprocessor.numeric_features = bundle.numeric_features
            self.preprocessor.categorical_features = bundle.categorical_features
        
        # For prediction, we need to ensure failure_probability exists in the input
        # (it will be dropped during transform, but some steps might check for it)
        if "failure_probability" not in df.columns:
            # Add a dummy column if missing (it will be dropped anyway)
            df = df.copy()
            df["failure_probability"] = 0.0
        
        X = self.preprocessor.transform_new(df)
        
        # Ensure exact feature match with training data
        if self.expected_feature_names is not None:
            # Create a DataFrame with the exact expected features
            X_aligned = pd.DataFrame(index=X.index)
            for feat_name in self.expected_feature_names:
                if feat_name in X.columns:
                    X_aligned[feat_name] = X[feat_name].values
                else:
                    # Fill missing features with 0
                    X_aligned[feat_name] = 0.0
            X = X_aligned[self.expected_feature_names]
        else:
            # Fallback: use the order from preprocessor
            feature_order = self.preprocessor.numeric_features + self.preprocessor.categorical_features
            X = X[feature_order]
        
        # Convert to numpy array to avoid feature name warnings
        return X.values

    def _explain_prediction(self, asset_dict: Dict[str, Any], feature_values: np.ndarray, probability: float) -> Dict[str, Any]:
        """Generate human-readable explanation for failure prediction."""
        if self.feature_importance is None or self.expected_feature_names is None:
            return {
                "reasons": [],
                "summary": "Model explanation not available"
            }
        
        # Map feature values to feature names
        feature_dict = {}
        if len(feature_values) == len(self.expected_feature_names):
            for i, feat_name in enumerate(self.expected_feature_names):
                feature_dict[feat_name] = float(feature_values[i])
        
        # Get actual operational values from asset_dict
        temp = self._safe_float(asset_dict.get("temperature"))
        vibration = self._safe_float(asset_dict.get("vibration_level"))
        pressure = self._safe_float(asset_dict.get("pressure"))
        runtime_hours = self._safe_float(asset_dict.get("runtime_hours"))
        
        # Calculate equipment age if not directly available
        equipment_age = self._safe_float(asset_dict.get("equipment_age_days"))
        if equipment_age is None and "install_date" in asset_dict:
            try:
                from datetime import datetime
                install_date_str = str(asset_dict.get("install_date", ""))
                if install_date_str:
                    # Try to parse date (format: "DD MM YYYY")
                    install_date = datetime.strptime(install_date_str, "%d %m %Y")
                    equipment_age = (datetime.now() - install_date).days
            except:
                equipment_age = None
        
        model_status = str(asset_dict.get("model_status", "")).strip()
        
        # Calculate contribution scores and identify issues
        contributions = []
        reasons_list = []
        
        # Check if temperature is significantly above average (threshold: q75 or mean + 1 std)
        if temp is not None and "temperature" in self.feature_importance:
            importance = self.feature_importance.get("temperature", 0)
            if self.training_stats and "temperature" in self.training_stats:
                stats = self.training_stats["temperature"]
                threshold = max(stats.get("q75", stats["mean"] + stats["std"]), stats["mean"] + stats["std"])
                if temp > threshold:
                    reasons_list.append(f"High temperature ({temp:.1f}°C)")
                    contributions.append({
                        "reason": f"High temperature ({temp:.1f}°C)",
                        "importance": importance,
                        "severity": "high" if temp > stats["mean"] + 2 * stats["std"] else "medium"
                    })
        
        # Check if vibration is significantly above average
        if vibration is not None and "vibration_level" in self.feature_importance:
            importance = self.feature_importance.get("vibration_level", 0)
            if self.training_stats and "vibration_level" in self.training_stats:
                stats = self.training_stats["vibration_level"]
                threshold = max(stats.get("q75", stats["mean"] + stats["std"]), stats["mean"] + stats["std"])
                if vibration > threshold:
                    reasons_list.append(f"Elevated vibration ({vibration:.2f} mm/s)")
                    contributions.append({
                        "reason": f"Elevated vibration ({vibration:.2f} mm/s)",
                        "importance": importance,
                        "severity": "high" if vibration > stats["mean"] + 2 * stats["std"] else "medium"
                    })
        
        # Check if pressure is significantly above average
        if pressure is not None and "pressure" in self.feature_importance:
            importance = self.feature_importance.get("pressure", 0)
            if self.training_stats and "pressure" in self.training_stats:
                stats = self.training_stats["pressure"]
                threshold = max(stats.get("q75", stats["mean"] + stats["std"]), stats["mean"] + stats["std"])
                if pressure > threshold:
                    reasons_list.append(f"High pressure ({pressure:.1f} bar)")
                    contributions.append({
                        "reason": f"High pressure ({pressure:.1f} bar)",
                        "importance": importance,
                        "severity": "high" if pressure > stats["mean"] + 2 * stats["std"] else "medium"
                    })
        
        # Check if runtime hours are significantly above average
        if runtime_hours is not None and "runtime_hours" in self.feature_importance:
            importance = self.feature_importance.get("runtime_hours", 0)
            if self.training_stats and "runtime_hours" in self.training_stats:
                stats = self.training_stats["runtime_hours"]
                threshold = max(stats.get("q75", stats["mean"] + stats["std"]), stats["mean"] + stats["std"])
                if runtime_hours > threshold:
                    reasons_list.append(f"Excessive runtime ({runtime_hours:.0f} hours)")
                    contributions.append({
                        "reason": f"Excessive runtime ({runtime_hours:.0f} hours)",
                        "importance": importance,
                        "severity": "medium"
                    })
        
        # Check if equipment age is significantly above average
        if equipment_age is not None and "equipment_age_days" in self.feature_importance:
            importance = self.feature_importance.get("equipment_age_days", 0)
            if self.training_stats and "equipment_age_days" in self.training_stats:
                stats = self.training_stats["equipment_age_days"]
                threshold = max(stats.get("q75", stats["mean"] + stats["std"]), stats["mean"] + stats["std"])
                if equipment_age > threshold:
                    reasons_list.append(f"Old equipment ({equipment_age:.0f} days old)")
                    contributions.append({
                        "reason": f"Old equipment ({equipment_age:.0f} days old)",
                        "importance": importance,
                        "severity": "medium"
                    })
        
        # Check model status (obsolete)
        if "model_status_Obsolete" in feature_dict and feature_dict["model_status_Obsolete"] > 0.5:
            importance = self.feature_importance.get("model_status_Obsolete", 0)
            reasons_list.append("Obsolete model - no longer supported")
            contributions.append({
                "reason": "Obsolete model - no longer supported",
                "importance": importance,
                "severity": "high"
            })
        
        # Check high pressure flag (verify against stricter threshold)
        if "high_pressure_flag" in feature_dict and feature_dict["high_pressure_flag"] > 0.5:
            # Double-check against stricter threshold
            if pressure is not None and self.training_stats and "pressure" in self.training_stats:
                stats = self.training_stats["pressure"]
                threshold = max(stats.get("q75", stats["mean"] + stats["std"]), stats["mean"] + stats["std"])
                if pressure > threshold:
                    importance = self.feature_importance.get("high_pressure_flag", 0)
                    if "High pressure" not in " ".join(reasons_list):
                        reasons_list.append("High pressure condition detected")
                        contributions.append({
                            "reason": "High pressure condition detected",
                            "importance": importance,
                            "severity": "high"
                        })
        
        # Check high temperature flag (verify against stricter threshold)
        if "high_temperature_flag" in feature_dict and feature_dict["high_temperature_flag"] > 0.5:
            # Double-check against stricter threshold
            if temp is not None and self.training_stats and "temperature" in self.training_stats:
                stats = self.training_stats["temperature"]
                threshold = max(stats.get("q75", stats["mean"] + stats["std"]), stats["mean"] + stats["std"])
                if temp > threshold:
                    importance = self.feature_importance.get("high_temperature_flag", 0)
                    if "High temperature" not in " ".join(reasons_list):
                        reasons_list.append("High temperature condition detected")
                        contributions.append({
                            "reason": "High temperature condition detected",
                            "importance": importance,
                            "severity": "high"
                        })
        
        # Check thermal stress indicator
        if "thermal_stress" in feature_dict and "thermal_stress" in self.feature_importance:
            importance = self.feature_importance.get("thermal_stress", 0)
            stress_value = feature_dict["thermal_stress"]
            if self.training_stats and "thermal_stress" in self.training_stats:
                stats = self.training_stats["thermal_stress"]
                if stress_value > stats["mean"]:
                    reasons_list.append("Elevated thermal stress")
                    contributions.append({
                        "reason": "Elevated thermal stress",
                        "importance": importance,
                        "severity": "high"
                    })
        
        # Check temperature-pressure stress indicator
        if "temp_pressure_stress" in feature_dict and "temp_pressure_stress" in self.feature_importance:
            importance = self.feature_importance.get("temp_pressure_stress", 0)
            if self.training_stats and "temp_pressure_stress" in self.training_stats:
                stats = self.training_stats["temp_pressure_stress"]
                stress_value = feature_dict["temp_pressure_stress"]
                if stress_value > stats["mean"]:
                    reasons_list.append("High temperature-pressure stress")
                    contributions.append({
                        "reason": "High temperature-pressure stress",
                        "importance": importance,
                        "severity": "high"
                    })
        
        # Sort by importance and severity
        contributions.sort(key=lambda x: (x["severity"] == "high", x["importance"]), reverse=True)
        
        # Get top reasons (limit to 5)
        top_reasons = [c["reason"] for c in contributions[:5]]
        
        # Don't show reasons for low-risk machines operating normally
        predicted_failure = probability >= self.prediction_threshold
        is_low_risk = probability < 0.50
        if is_low_risk and not predicted_failure:
            top_reasons = []
        
        # If no specific reasons found, handle based on probability
        # For high-risk machines, always provide at least generic reasons
        if not top_reasons:
            if probability >= 0.85:
                top_reasons = ["Multiple critical risk factors detected", "High failure probability"]
            elif probability >= 0.70:
                top_reasons = ["High-risk operational conditions", "Elevated failure probability"]
            elif probability >= 0.50:
                top_reasons = ["Some risk factors identified"]
            # For low-risk machines, we already cleared reasons above, so keep empty
        # Ensure high-risk machines always have reasons even if generic
        elif probability >= 0.70 and len(top_reasons) < 2:
            # Add generic reason if we have very few specific ones
            if probability >= 0.85:
                top_reasons.append("Critical failure risk")
            else:
                top_reasons.append("Elevated failure probability")
        
        # Create summary
        if probability >= 0.85:
            summary = "Critical risk factors detected"
        elif probability >= 0.70:
            summary = "Multiple high-risk factors present"
        elif probability >= 0.50:
            summary = "Some risk factors identified"
        else:
            summary = "Low risk - normal operation"
        
        return {
            "reasons": top_reasons,
            "summary": summary
        }
    
    @staticmethod
    def _calculate_maintenance_days(probability: float) -> int:
        """Calculate recommended maintenance days based on failure probability."""
        import random
        random.seed(int(probability * 1000))
        if probability >= 0.85:
            return max(1, 2 + random.randint(-1, 1))
        elif probability >= 0.70:
            base_days = int(7 - (probability - 0.70) / 0.15 * 4)
            return max(2, base_days + random.randint(-1, 1))
        elif probability >= 0.50:
            base_days = int(18 - (probability - 0.50) / 0.20 * 8)
            return max(8, base_days + random.randint(-2, 2))
        else:
            base_days = int(45 - probability / 0.50 * 20)
            return max(20, base_days + random.randint(-3, 3))
    
    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Safely convert value to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability >= 0.85:
            return "very_high"
        if probability >= 0.70:
            return "high"
        if probability >= 0.50:
            return "medium"
        return "low"
