# Plant Maintenance ML Pipeline

A simple machine learning pipeline to predict equipment failures in industrial plants.

## What it does

This project helps predict which equipment might fail soon based on sensor data like vibration, temperature, pressure, and maintenance history.

## Files

- `plant_dataset.csv` - Your equipment data (sensor readings, maintenance dates, etc.)
- `data_load_and_eda.py` - Explores the dataset and shows basic statistics
- `preprocess_and_features.py` - Cleans and prepares the data for training
- `train_models.py` - Trains a model to predict failures
- `utils.py` - Helper functions for loading data and saving models

## How to use

1. **Prepare your data**
   ```bash
   python preprocess_and_features.py
   ```
   This creates `processed_data.csv` with clean, numeric features.

2. **Train the model**
   ```bash
   python train_models.py --csv plant_dataset.csv
   ```
   This creates:
   - `rf_model.joblib` - The trained model
   - `encoders.joblib` - How to process new data
   - `label_encoder_target.joblib` - Maps predictions back to readable labels

## What the model predicts

By default, it predicts `is_failure_soon`:
- **0** = Equipment is fine, no immediate failure risk
- **1** = Equipment might fail soon, needs attention

The model automatically creates this target from your `failure_probability` column.

## Example results

```
Accuracy: 0.8636
Class distribution: {'0': 52, '1': 54}

              precision    recall  f1-score   support
           0       0.83      0.91      0.87        11
           1       0.90      0.82      0.86        11
```

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- joblib

## Notes

- The model uses a simple RandomForest with 100 trees
- Numeric columns (like temperature, pressure) keep their original values
- Text columns are converted to numbers using smart encoding
- The model prevents data leakage by excluding the target column from features
