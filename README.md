# Plant Maintenance Q&A System

This project lets reliability engineers ask natural‑language questions about plant equipment and immediately get answers backed by both the historical dataset and a trained ML model. It ships with two front doors:

- **Streamlit app (`app.py`)** – interactive UI with example questions and formatted answers.
- **FastAPI service (`plant_server.py`)** – REST endpoint for integrating with other tools.

---

## Project Highlights

- **Unified question workflow**: Parse → analyze → format so every surface (CLI, API, Streamlit) behaves the same.
- **ML-backed insights**: ExtraTrees model predicts failure probability, risk level, reasons, and maintenance timing.
- **Top-N views**: Quickly list high-risk, low-risk, or preventive candidates without showing raw probabilities.
- **User-friendly explanations**: Low-risk machines stay calm, high-risk machines get detailed reasons, all phrased like a human wrote them.
- **Production-ready API**: FastAPI + Pydantic responses, CORS enabled, clean error messages.

---

## Directory Guide

| Path | Description |
|------|-------------|
| `app.py` | Streamlit UI with categorized example questions, automatic reruns, and neatly formatted answers. |
| `plant_server.py` | FastAPI REST service exposing `/ask`. Loads parser, analyzer, and formatter on startup. |
| `question_parser.py` | Regex-driven intent detector for counts, averages, top-N, predictive/risk questions, etc. |
| `data_analyzer.py` | Core orchestration layer; loads data, calls the ML predictor, and builds structured results. |
| `answer_formatter.py` | Converts analysis dictionaries into conversational paragraphs with reasons and maintenance info. |
| `ml_predictor.py` | Loads the trained model + preprocessing artifacts, generates probabilities, reasons, and schedules. |
| `ml_training/` | Training pipeline (preprocessing, model selection, artifacts, and evaluation reports). |
| `plant_dataset.csv` | Sample dataset used for both analysis and training. |

---

## Getting Started

### 1. Environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Run the Streamlit UI
```bash
streamlit run app.py
```
- Use the sidebar example questions (dataset queries + ML predictions).
- Click any example to auto-populate the input and re-run.

### 3. Run the FastAPI Server
```bash
python plant_server.py
# or
uvicorn plant_server:app --reload --host 0.0.0.0 --port 8000
```

Send questions via `POST /ask`:
```bash
curl -X POST http://localhost:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"Will COMP-017 fail soon?\"}"
```

---

## Example Predictive Questions
- “Will COMP-017 fail soon?”
- “Predict the failure probability for TRANS-044.”
- “Top 5 high-risk machines for preventive maintenance.”
- “Show top 3 low-risk assets.”
- “When should I schedule maintenance for MOTOR-012?”

Dataset-style questions still work, e.g., “How many pumps are there?” or “What’s the average temperature?”

---

## Architecture Overview

1. **QuestionParser**  
   Cleans the sentence, detects intent (count, top, predictive, etc.), pulls out parameters like asset IDs or requested risk levels.

2. **DataAnalyzer**  
   - Loads/caches `plant_dataset.csv`.  
   - Routes the parsed intent to the right handler.  
   - Calls `MaintenancePredictor` when ML insight is needed.  
   - Packages results with risk levels, maintenance days, and provenance.

3. **MaintenancePredictor (ml_predictor.py)**  
   - Recreates the training preprocessing pipeline.  
   - Runs the ExtraTrees model to get failure probability.  
   - Generates human-readable reasons using training statistics.  
   - Assigns maintenance timing with probability-based jitter.

4. **AnswerFormatter**  
   - Turns analyzer dicts into paragraphs.  
   - Ensures low-risk equipment shows “All parameters within normal range,” while risky assets list concrete issues.  
   - Used by both Streamlit and FastAPI for consistent tone.

---

## Training Pipeline (`ml_training/`)

- `data_preprocessing.py` – feature engineering (one-hot + frequency encoding, scaling, feature selection).
- `model_trainer.py` – trains several models, tunes thresholds, saves metrics.
- `train_plant_maintenance.py` – CLI entry point for retraining, outputs artifacts + reports.
- `artifacts.json` – serialized preprocessing stats (means, stds, thresholds, selected features).
- `models/` – trained model (`model.joblib`), feature importance, leaderboard, evaluation report.

To retrain:
```bash
python ml_training/train_plant_maintenance.py
```
It refreshes `models/` and `artifacts.json`, which the predictor automatically picks up.

---

## Testing & Validation Tips

- **Parser coverage**: Try odd phrasings (“show me low-risk valves”) to confirm the fallback guidance works.
- **Predictor sanity**: Spot-check a few assets via `ml_predictor.py` or Streamlit to ensure reasons/risk match expectations.
- **API smoke test**: `curl` or Postman against `/ask` to verify JSON responses.
- **Streamlit UI**: Use the example buttons and free-form input to confirm reruns and formatting.

---

## Troubleshooting

- **No answer / blank Streamlit output** → Ensure `plant_dataset.csv` is present and Session State buttons were clicked (they auto-rerun).
- **Reasons showing for low-risk machines** → Confirm you’re using the latest `ml_predictor.py` (it suppresses reasons when `probability < 0.5`).
- **Parser says “rephrase”** → The question didn’t match any intent; add more detail (asset ID, action, or metric).
- **Training mismatch** → Re-run `train_plant_maintenance.py` so `artifacts.json` and `model.joblib` are aligned.

---
