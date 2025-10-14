# Plant Maintenance ML Pipeline

A simple machine learning pipeline to predict equipment failures in industrial plants.

## What it does

This project helps predict which equipment might fail soon based on sensor data like vibration, temperature, pressure, and maintenance history.

## Files

- `plant_dataset.csv` - Your equipment data (sensor readings, maintenance dates, etc.)
- `preprocess_and_features.py` - Cleans and prepares the data for training
- `train_models.py` - Trains a model to predict failures
- `utils.py` - Helper functions for loading data, saving models, and data exploration

## How to use

1. **Explore your data** (optional)
   ```bash
   python utils.py
   ```
   This shows basic statistics and data exploration for the plant dataset.

2. **Prepare your data**
   ```bash
   python preprocess_and_features.py
   ```
   This creates `processed_data.csv` with clean, numeric features.

3. **Train the model**
   ```bash
   python train_models.py
   ```
   This creates:
   - `rf_model.joblib` - The trained model
   - `encoders.joblib` - How to process new data
   - `label_encoder_target.joblib` - Maps predictions back to readable labels

## Configuration (no command-line args)

The training script uses fixed defaults. To change them, edit these constants at the top of `train_models.py`:

```python
CSV_PATH = "plant_dataset.csv"
TARGET_COL = "is_failure_soon"
FAILURE_THRESHOLD = 0.5
MODEL_OUT = "rf_model.joblib"
ENCODERS_OUT = "encoders.joblib"
LABEL_OUT = "label_encoder_target.joblib"
```

- Change `CSV_PATH` to point to a different dataset file.
- If your CSV doesn’t have `is_failure_soon`, it will be derived from `failure_probability >= FAILURE_THRESHOLD`.

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
