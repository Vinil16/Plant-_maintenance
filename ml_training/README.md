# ML Training Module

This directory contains all machine learning training code and models, kept separate from the main Q&A system.

## 📁 Structure

```
ml_training/
├── README.md                 # This file
├── train_models.py          # Main training script
├── preprocess_and_features.py # Data preprocessing
├── utils.py                 # Utility functions
├── models/                  # Trained models
│   ├── rf_model.joblib     # RandomForest model
│   ├── encoders.joblib     # Feature encoders
│   └── label_encoder_target.joblib # Target encoder
├── results/                 # Training results
│   └── model_results/      # Model performance results
└── requirements.txt         # ML-specific dependencies
```

## 🎯 Purpose

- **Separate ML concerns** from Q&A system
- **Keep training code organized** and maintainable
- **Allow independent ML development** without affecting Q&A
- **Easy to retrain models** when needed

## 🚀 Usage

### Training a New Model
```bash
cd ml_training
python train_models.py
```

### Using Trained Models
The Q&A system can optionally use these models for predictions, but works fine without them.

## 📊 Model Information

- **Algorithm:** RandomForest Classifier
- **Target:** is_failure_soon (binary classification)
- **Features:** 20+ engineered features from plant data
- **Performance:** 96%+ accuracy with cross-validation
- **Purpose:** Predict equipment failure risk

## 🔧 Dependencies

See `requirements.txt` for ML-specific packages:
- scikit-learn
- pandas
- numpy
- joblib


