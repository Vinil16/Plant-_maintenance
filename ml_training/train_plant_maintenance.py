"""
Simple script to train predictive maintenance models.

This script:
1. Preprocesses the raw plant dataset
2. Trains multiple ML models and picks the best one
3. Saves everything to the models folder

Just run: python train_plant_maintenance.py
"""

import json
import subprocess
import sys
from pathlib import Path

from data_preprocessing import DataPreprocessor


def main():
    """Main training pipeline: preprocess data and train models."""
    print("\n" + "=" * 70)
    print("Plant Maintenance - ML Training Pipeline")
    print("=" * 70)
    
    # Setup paths
    base_dir = Path(__file__).parent
    data_file = base_dir.parent / "plant_dataset.csv"
    output_dir = base_dir / "models"
    processed_file = base_dir / "processed_dataset.csv"
    
    # Step 1: Preprocess data
    print("\nStep 1: Preprocessing data...")
    preprocessor = DataPreprocessor(
        data_path=data_file,
        target_name="is_fail_soon",
        threshold=0.710,
        scaler="standard"
    )
    bundle = preprocessor.prepare(save_processed=True, output_dir=str(base_dir))
    print(f"  Samples: {len(bundle.features)}, Features: {len(bundle.feature_names)}")
    
    # Step 2: Train models
    print("\nStep 2: Training models...")
    
    output_dir.mkdir(exist_ok=True)
    trainer_script = base_dir / "model_trainer.py"
    
    if not trainer_script.exists():
        print(f"Trainer script not found: {trainer_script}")
        return
    
    # Run the model trainer
    cmd = [
        sys.executable,
        str(trainer_script),
        "--data", str(processed_file),
        "--out_dir", str(output_dir),
        "--target-col", "target",
        "--seed", "42"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        
        # Show final results
        report_file = output_dir / "eval_report.json"
        if report_file.exists():
            with open(report_file) as f:
                report = json.load(f)
            print("\n" + "=" * 70)
            print("Final Results")
            print("=" * 70)
            print(f"Best Model: {report['best_model']}")
            print(f"Accuracy:  {report['val_metrics']['accuracy']:.4f}")
            print(f"ROC AUC:   {report['val_metrics']['roc_auc']:.4f}")
            print(f"F1 Score:  {report['val_metrics']['f1']:.4f}")
            print(f"\nAll results saved to: {output_dir}")
            print("=" * 70)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
        if e.stderr:
            print("Errors:", e.stderr)
        return
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return
    
    print("\nPipeline finished successfully")


if __name__ == "__main__":
    main()
