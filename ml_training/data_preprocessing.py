# data_preprocessor.py
"""
Predictive Maintenance — Data Preprocessor

- Creates binary target is_fail_soon from failure_probability >= 0.710 (no leakage)
- Cleans data, engineers time/ops features
- Encodes categoricals (one-hot for low-card, frequency for high-card)
- Scales numeric features (standard or minmax, or disable)
- Saves artifacts for consistent inference

Usage:
    from data_preprocessor import DataPreprocessor
    dp = DataPreprocessor(data_path="plant_dataset.csv", scaler="standard")  # "standard" | "minmax" | None
    bundle = dp.prepare(save_processed=True, output_dir="ml_training")
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd


TARGET_THRESHOLD = 0.710
LOW_CARD_MAX = 15          # one-hot if unique <= 15
TOP_N_ASSET_TYPES = 6      # one-hot only top-N asset types to keep feature count sane


@dataclass
class DatasetBundle:
    features: pd.DataFrame
    target: pd.Series
    numeric_features: List[str]
    categorical_features: List[str]
    feature_names: List[str]
    encoder_info: Dict[str, dict]
    scaling_info: Dict[str, Dict[str, float]]


class DataPreprocessor:
    """Preprocesses plant maintenance data for ML training."""
    def __init__(
        self,
        data_path: str | Path,
        failure_score_column: str = "failure_probability",
        target_name: str = "is_fail_soon",
        threshold: float = TARGET_THRESHOLD,
        scaler: Optional[str] = "standard",   # "standard" | "minmax" | None
    ) -> None:
        self.data_path = Path(data_path)
        self.failure_score_column = failure_score_column
        self.target_name = target_name
        self.threshold = threshold
        self.scaler_mode = scaler
        self._raw: Optional[pd.DataFrame] = None
        # Learned during prepare()
        self.one_hot_columns: List[str] = []
        self.one_hot_domains: Dict[str, List[str]] = {}
        self.freq_maps: Dict[str, Dict[str, float]] = {}
        self.scaling_stats: Dict[str, Dict[str, float]] = {"mean": {}, "std": {}, "min": {}, "max": {}}
        self.low_variance_features: List[str] = []
        self.high_corr_drop: List[str] = []
        self.freq_smoothing: float = 1.0
        # Final feature lists
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []

    # ---------- Public API ----------

    def load(self) -> pd.DataFrame:
        """Load raw dataset from CSV file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path.resolve()}")
        df = pd.read_csv(self.data_path, encoding="latin1")
        df.columns = [c.strip() for c in df.columns]
        self._raw = df
        return df

    def prepare(self, save_processed: bool = True, output_dir: str | Path | None = None) -> DatasetBundle:
        """Prepare dataset: clean, engineer features, encode, scale, and save artifacts."""
        df = self._ensure_loaded().copy()
        df = self._clean(df)
        df = self._engineer_features(df)
        y = self._build_target(df)
        X = self._encode_categoricals(df)
        X = self._select_and_impute_numeric(X)
        X = self._scale_numeric(X)
        X = self._postprocess_features(X)
        # Record feature lists in the learned order
        feature_names = list(X.columns)
        self.numeric_features = [c for c in feature_names if c in self.numeric_features]
        self.categorical_features = [c for c in feature_names if c not in self.numeric_features]

        if save_processed:
            # Save in ml_training folder (same folder as this script)
            out_dir = Path(output_dir) if output_dir else (Path(__file__).parent)
            out_dir.mkdir(parents=True, exist_ok=True)
            # save dataset
            out_df = X.copy()
            out_df["target"] = y.astype(int).values
            out_path = out_dir / "processed_dataset.csv"
            out_df.to_csv(out_path, index=False)
            print(f"Saved processed dataset to {out_path}")
            # save artifacts
            artifacts = {
                "target_name": self.target_name,
                "threshold": self.threshold,
                "scaler_mode": self.scaler_mode,
                "numeric_features": self.numeric_features,
                "categorical_features": self.categorical_features,
                "one_hot_domains": self.one_hot_domains,
                "scaling_stats": self.scaling_stats,
                "low_variance_features": self.low_variance_features,
                "high_corr_drop": self.high_corr_drop,
            }
            (out_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2))

        return DatasetBundle(
            features=X,
            target=y,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            feature_names=list(X.columns),
            encoder_info={"one_hot_domains": self.one_hot_domains},
            scaling_info=self.scaling_stats,
        )

    def transform_new(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the *fitted* preprocessing to new data for inference.
        Assumes `prepare()` has run already (artifacts are in memory).
        """
        if self._raw is None:
            raise RuntimeError("Call prepare() first to fit encoders & scalers.")

        df = self._clean(df.copy())
        df = self._engineer_features(df)
        X = self._encode_categoricals(df, training=False)
        X = self._select_and_impute_numeric(X, training=False)
        X = self._scale_numeric(X, training=False)
        X = self._postprocess_features(X, training=False)

        # ensure exact training column order / fill missing
        for col in self.numeric_features + self.categorical_features:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.numeric_features + self.categorical_features]
        return X

    # ---------- Steps ----------

    def _ensure_loaded(self) -> pd.DataFrame:
        return self._raw if self._raw is not None else self.load()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data: replace missing value strings and remove duplicates."""
        df = df.replace({"": np.nan, "NA": np.nan, "N/A": np.nan, "nan": np.nan, "None": np.nan})
        if "asset_id" in df.columns:
            df = df.drop_duplicates(subset="asset_id", keep="last")
        return df

    def _build_target(self, df: pd.DataFrame) -> pd.Series:
        """Create binary target from failure probability threshold."""
        if self.failure_score_column not in df.columns:
            raise ValueError(f"Missing '{self.failure_score_column}' column for target creation.")
        scores = pd.to_numeric(df[self.failure_score_column], errors="coerce")
        y = (scores >= self.threshold).astype(int)
        return y

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer time-based, operational, and interaction features."""
        cur = pd.Timestamp.now().normalize()
        out = df.copy()
        # Date-based features
        to_dt = lambda s: pd.to_datetime(s, errors="coerce", dayfirst=True)
        if "install_date" in out.columns:
            age = (cur - to_dt(out["install_date"])).dt.days
            out["equipment_age_days"] = age.clip(lower=0)
        else:
            out["equipment_age_days"] = np.nan
        if "last_maintenance_date" in out.columns:
            last = (cur - to_dt(out["last_maintenance_date"])).dt.days
            out["days_since_last_maintenance"] = last.clip(lower=0)
        else:
            out["days_since_last_maintenance"] = np.nan
        if "next_due_date" in out.columns:
            until = (to_dt(out["next_due_date"]) - cur).dt.days
            out["days_until_next_maintenance"] = until
        else:
            out["days_until_next_maintenance"] = np.nan
        out["is_maintenance_overdue"] = (out["days_until_next_maintenance"] < 0).astype(int)
        if "model_release_year" in out.columns:
            mry = pd.to_numeric(out["model_release_year"], errors="coerce")
            out["model_age_years"] = (cur.year - mry).clip(lower=0)
        else:
            out["model_age_years"] = np.nan
        # Operational sensor features
        for col in ["runtime_hours", "vibration_level", "temperature", "pressure"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
            else:
                out[col] = np.nan
        # Engineered features
        out["runtime_intensity"] = out["runtime_hours"] / (out["equipment_age_days"].fillna(0) + 1)
        out["maintenance_frequency"] = out["runtime_hours"] / (out["days_since_last_maintenance"].fillna(0) + 1)
        # Temperature-pressure interaction features
        if "temperature" in out.columns and "pressure" in out.columns:
            temp_norm = pd.to_numeric(out["temperature"], errors="coerce")
            press_norm = pd.to_numeric(out["pressure"], errors="coerce")
            out["temp_pressure_stress"] = temp_norm * press_norm
            out["temp_pressure_ratio"] = temp_norm / (press_norm + 1e-6)
            temp_median = temp_norm.median()
            out["high_temperature_flag"] = (temp_norm > temp_median).astype(int)
            press_median = press_norm.median()
            out["high_pressure_flag"] = (press_norm > press_median).astype(int)
            out["thermal_stress"] = (temp_norm > temp_median).astype(int) + (press_norm > press_median).astype(int)
        # Temperature-runtime interaction
        if "temperature" in out.columns and "runtime_hours" in out.columns:
            temp_val = pd.to_numeric(out["temperature"], errors="coerce")
            runtime_val = pd.to_numeric(out["runtime_hours"], errors="coerce")
            out["temp_per_runtime"] = temp_val / (runtime_val + 1)
        # Pressure-runtime interaction
        if "pressure" in out.columns and "runtime_hours" in out.columns:
            press_val = pd.to_numeric(out["pressure"], errors="coerce")
            runtime_val = pd.to_numeric(out["runtime_hours"], errors="coerce")
            out["pressure_per_runtime"] = press_val / (runtime_val + 1)
        # Vibration-temperature interaction
        if "vibration_level" in out.columns and "temperature" in out.columns:
            vib_val = pd.to_numeric(out["vibration_level"], errors="coerce")
            temp_val = pd.to_numeric(out["temperature"], errors="coerce")
            out["vib_temp_interaction"] = vib_val * temp_val
        return out

    def _encode_categoricals(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """Encode categorical features: one-hot for low-cardinality, frequency for high-cardinality."""
        X = df.copy()
        cats_low = []
        if "model_status" in X.columns:
            cats_low.append("model_status")
        # Asset type: one-hot encode top-N categories
        if "asset_type" in X.columns:
            X["asset_type"] = X["asset_type"].fillna("Unknown")
            if training:
                top = X["asset_type"].value_counts().head(TOP_N_ASSET_TYPES).index.tolist()
                self.one_hot_domains["asset_type"] = top
            top = self.one_hot_domains.get("asset_type", [])
            for val in top:
                col = f"asset_type_{val}"
                X[col] = (X["asset_type"] == val).astype(int)
            X = X.drop(columns=["asset_type"])
        # One-hot encode low-cardinality categoricals
        for col in cats_low:
            X[col] = X[col].fillna("Unknown").astype(str)
            if training:
                dom = sorted(X[col].unique().tolist())
                self.one_hot_domains[col] = dom
            dom = self.one_hot_domains.get(col, [])
            for val in dom:
                newc = f"{col}_{val}"
                X[newc] = (X[col] == val).astype(int)
            X = X.drop(columns=[col])
        # Frequency encoding for high-cardinality categoricals
        for hc in ["installed_model", "latest_model", "location", "asset_family"]:
            if hc in X.columns:
                series = X[hc].astype(str).fillna("Unknown")
                if training:
                    counts = series.value_counts(dropna=False)
                    total = float(counts.sum())
                    smoothing = self.freq_smoothing
                    freq = (counts + smoothing) / (total + smoothing * len(counts))
                    mapping = freq.to_dict()
                    mapping["__default__"] = float(smoothing / (total + smoothing * len(counts))) if total else 0.0
                    self.freq_maps[hc] = mapping
                mapping = self.freq_maps.get(hc, {})
                default_val = mapping.get("__default__", 0.0)
                applied_map = {k: v for k, v in mapping.items() if k != "__default__"}
                X[f"{hc}_freq"] = series.map(applied_map).fillna(default_val).astype(float)
                X = X.drop(columns=[hc])
        return X

    def _select_and_impute_numeric(self, X: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """Select numeric features and impute missing values with median."""
        numeric_order = [
            "vibration_level", "temperature", "pressure", "runtime_hours",
            "equipment_age_days", "days_since_last_maintenance",
            "days_until_next_maintenance", "model_age_years",
            "runtime_intensity", "maintenance_frequency",
            "is_maintenance_overdue",
            "temp_pressure_stress", "temp_pressure_ratio",
            "high_temperature_flag", "high_pressure_flag", "thermal_stress",
            "temp_per_runtime", "pressure_per_runtime",
            "vib_temp_interaction",
        ]
        present = [c for c in numeric_order if c in X.columns]
        # Impute missing values with median
        for c in present:
            if X[c].dtype.kind not in "fcbiu":
                X[c] = pd.to_numeric(X[c], errors="coerce")
            med = float(X[c].median()) if not np.isnan(X[c].median()) else 0.0
            X[c] = X[c].fillna(med)
        self.numeric_features = present
        # Drop non-numeric columns that shouldn't be in the model
        non_model_cols = [
            c for c in X.columns
            if c not in self.numeric_features
            and c != self.failure_score_column
            and X[c].dtype.kind not in "fcbiu"
        ]
        if non_model_cols:
            X = X.drop(columns=non_model_cols)
        # Remember one-hot/encoded columns
        self.categorical_features = [
            c for c in X.columns
            if c not in self.numeric_features and c != self.failure_score_column
        ]
        # Drop redundant or leaky columns
        drop_cols = [self.failure_score_column, "model_release_year", "install_date", "last_maintenance_date", "next_due_date"]
        X = X.drop(columns=drop_cols, errors="ignore")
        return X

    def _postprocess_features(self, X: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """Remove low-variance and highly correlated features."""
        if training:
            variances = X.var(axis=0)
            self.low_variance_features = variances[variances < 1e-6].index.tolist()
            if self.low_variance_features:
                X = X.drop(columns=self.low_variance_features, errors="ignore")
            if X.shape[1] > 1:
                corr = X.corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                self.high_corr_drop = [
                    column
                    for column in upper.columns
                    if any(upper[column] > 0.98)
                ]
                if self.high_corr_drop:
                    X = X.drop(columns=self.high_corr_drop, errors="ignore")
            else:
                self.high_corr_drop = []
        else:
            if self.low_variance_features:
                X = X.drop(columns=[c for c in self.low_variance_features if c in X.columns], errors="ignore")
            if self.high_corr_drop:
                X = X.drop(columns=[c for c in self.high_corr_drop if c in X.columns], errors="ignore")
        return X

    def _scale_numeric(self, X: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """Scale numeric features using standard or minmax scaling."""
        if self.scaler_mode is None:
            return X
        if self.scaler_mode not in ("standard", "minmax"):
            raise ValueError("scaler must be 'standard', 'minmax', or None")
        X = X.copy()
        for c in self.numeric_features:
            col = X[c].astype(float)
            if training:
                if self.scaler_mode == "standard":
                    mu = float(col.mean()); sd = float(col.std(ddof=0)) or 1.0
                    self.scaling_stats["mean"][c] = mu
                    self.scaling_stats["std"][c] = sd
                    X[c] = (col - mu) / sd
                else:  # minmax
                    mn = float(col.min()); mx = float(col.max())
                    rng = (mx - mn) or 1.0
                    self.scaling_stats["min"][c] = mn
                    self.scaling_stats["max"][c] = mx
                    X[c] = (col - mn) / rng
            else:
                if self.scaler_mode == "standard":
                    mu = self.scaling_stats["mean"].get(c, 0.0)
                    sd = self.scaling_stats["std"].get(c, 1.0) or 1.0
                    X[c] = (col - mu) / sd
                else:
                    mn = self.scaling_stats["min"].get(c, 0.0)
                    mx = self.scaling_stats["max"].get(c, 1.0)
                    rng = (mx - mn) or 1.0
                    X[c] = (col - mn) / rng
        return X
