"""
Optional ML Predictor
Can be used by the Q&A system if ML predictions are needed
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

class MLPredictor:
    """
    Optional ML predictor that can be used by the Q&A system.
    Keeps ML functionality separate and optional.
    """
    
    def __init__(self, models_dir="models"):
        """Initialize with models directory."""
        self.models_dir = Path(models_dir)
        self.model = None
        self.encoders = None
        self.label_encoder = None
        self.loaded = False
        
        self._load_models()
    
    def _load_models(self):
        """Load trained models if available."""
        try:
            model_path = self.models_dir / "rf_model.joblib"
            encoders_path = self.models_dir / "encoders.joblib"
            label_path = self.models_dir / "label_encoder_target.joblib"
            
            if model_path.exists():
                self.model = joblib.load(model_path)
                print("✅ ML model loaded")
            
            if encoders_path.exists():
                self.encoders = joblib.load(encoders_path)
                print("✅ Encoders loaded")
            
            if label_path.exists():
                self.label_encoder = joblib.load(label_path)
                print("✅ Label encoder loaded")
            
            self.loaded = bool(self.model)
            
        except Exception as e:
            print(f"⚠️ Could not load ML models: {e}")
            self.loaded = False
    
    def is_available(self):
        """Check if ML models are available."""
        return self.loaded
    
    def predict_risk(self, asset_data):
        """
        Predict failure risk for asset data.
        
        Args:
            asset_data: dict or DataFrame with asset information
            
        Returns:
            dict with prediction results
        """
        if not self.loaded:
            return {"error": "ML models not available"}
        
        try:
            # Convert to DataFrame if needed
            if isinstance(asset_data, dict):
                asset_data = pd.DataFrame([asset_data])
            
            # Import preprocessing function
            from preprocess_and_features import preprocess_data
            
            # Preprocess data
            processed_data = preprocess_data(asset_data, target_col="is_failure_soon")
            
            # Get feature columns
            feature_cols = [col for col in processed_data.columns 
                           if col not in ['is_failure_soon', 'failure_probability', 'asset_id']]
            
            X = processed_data[feature_cols]
            
            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
            
            # Format results
            results = []
            for i, (idx, row) in enumerate(asset_data.iterrows()):
                prob = float(probabilities[i])
                results.append({
                    'asset_id': row['asset_id'],
                    'asset_type': row['asset_type'],
                    'predicted_failure': bool(predictions[i]),
                    'failure_probability': prob,
                    'risk_level': 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low',
                    'model_confidence': float(max(self.model.predict_proba(X)[i]))
                })
            
            return {"success": True, "predictions": results, "source": "ML Model"}
            
        except Exception as e:
            return {"error": f"ML prediction failed: {str(e)}"}

# Example usage
if __name__ == "__main__":
    predictor = MLPredictor()
    
    if predictor.is_available():
        print("✅ ML Predictor is ready")
        
        # Example asset data
        test_asset = {
            'asset_id': 'TEST-001',
            'asset_type': 'Test Pump',
            'runtime_hours': 2000,
            'vibration_level': 4.5,
            'temperature': 85,
            'pressure': 6.0
        }
        
        result = predictor.predict_risk(test_asset)
        if result.get('success'):
            pred = result['predictions'][0]
            print(f"Prediction: {pred['asset_id']} - {pred['failure_probability']:.3f} ({pred['risk_level']})")
    else:
        print("⚠️ ML Predictor not available - use dataset values instead")
