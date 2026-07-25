# Social Media Addiction Analyzer

> An end-to-end ML-powered web application that predicts addiction scores and productivity levels from social media usage patterns.

---

## 📁 Project Structure

```
social_media_analyzer/
│
├── data/
│   └── Students_Social_Media_Addiction.csv   # 705-student training dataset
│
├── ml/
│   ├── train_model.py                        # Train ML models (run first)
│   ├── app.py                                # Flask REST API server
│   └── models/                               # Auto-generated after training
│       ├── addiction_model.pkl               # Gradient Boosting Regressor
│       ├── productivity_model.pkl            # Random Forest Classifier
│       ├── dataset_stats.pkl                 # Aggregated stats for dashboard
│       ├── encoder_classes.pkl               # Valid input options
│       └── le_*.pkl                          # Label encoders
│
├── frontend/
│   └── index.html                            # Full web application (single file)
│
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

---

## 🚀 How to Run

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Train the ML Models

```bash
cd ml
python train_model.py
```

**Expected output:**

- Addiction Model: MAE ≈ 0.115, R² ≈ 0.98
- Productivity Model: Accuracy ≈ 96.5%

### Step 3 — Start the Flask API

```bash
python app.py
```

API will run at: `http://localhost:5000`

### Step 4 — Open the Frontend

Open `frontend/index.html` in any browser.

> ⚠️ The frontend works even if the API is offline — it falls back to a rule-based demo mode. For full ML predictions, the API must be running.

---

## 🤖 ML Models

### 1. Addiction Score Predictor

| Property   | Value                        |
|------------|-------------------------------|
| Algorithm  | Gradient Boosting Regressor  |
| Target     | `Addicted_Score` (1–9)       |
| MAE        | ~0.115                       |
| R² Score   | ~0.98                        |
| CV R²      | ~0.955 ± 0.024                |

**Top Features:**

1. Mental Health Score (82% importance)
2. Conflicts Over Social Media (11%)
3. Affects Academic Performance (3%)

### 2. Productivity Level Predictor

| Property   | Value                     |
|------------|----------------------------|
| Algorithm  | Random Forest Classifier |
| Target     | Low / Medium / High      |
| Accuracy   | ~96.5%                    |
| Classes    | 3 (Low, Medium, High)     |

**Productivity is derived from:**

```
Sleep × 1.5 + Mental Health × 1.2 − Usage × 1.0 − Conflicts × 0.5
```

---

## 📊 Dataset

- **Source:** `Students_Social_Media_Addiction.csv`
- **Size:** 705 students
- **Features:** Age, Gender, Academic Level, Country, Daily Usage (hours), Platform, Affects Academic Performance, Sleep Hours, Mental Health Score, Relationship Status, Conflicts Over Social Media
- **Target:** Addicted_Score (2–9)

### Key Dataset Findings

- 64.3% of students report social media affects academic performance
- Average addiction score: **6.44 / 9**
- Average daily usage: **4.19 hours**
- TikTok shows the **highest average addiction score** among platforms

---

## 🔗 Power BI Dashboard Integration

The Power BI `.pbix` file (`IDA_PROJECT.pbix`) visualizes the same dataset used for ML training.

**How they connect:**

| Power BI Dashboard                        | ML Analyzer                                  |
|--------------------------------------------|-----------------------------------------------|
| Shows aggregate trends (all 705 students)  | Predicts for your individual profile          |
| Platform-wise addiction patterns           | Feature importance from the same variables    |
| Sleep vs. addiction heatmaps               | Sleep is used as a key ML input feature       |
| Academic level comparisons                 | Academic level is an encoded ML feature       |

**To embed Power BI in the app:**

1. Open `IDA_PROJECT.pbix` in Power BI Desktop
2. Publish it to Power BI Service (powerbi.com)
3. Go to File → Embed Report → Website or Portal
4. Copy the embed URL
5. In the web app → click "Power BI Dashboard" → paste the URL → click "Load Dashboard"

---

## 🌐 API Endpoints

| Method | Endpoint         | Description                                     |
|--------|------------------|--------------------------------------------------|
| `POST` | `/api/analyze`   | Predict addiction score + productivity          |
| `GET`  | `/api/stats`     | Get dataset-level aggregated statistics         |
| `GET`  | `/api/options`   | Get valid input options (platforms, genders, etc.) |
| `GET`  | `/api/health`    | Health check                                    |

### Example POST `/api/analyze`

```json
{
  "age": 20,
  "gender": "Female",
  "academic_level": "Undergraduate",
  "avg_daily_usage_hours": 5.5,
  "most_used_platform": "Instagram",
  "affects_academic_performance": "Yes",
  "sleep_hours_per_night": 6.0,
  "mental_health_score": 5,
  "relationship_status": "Single",
  "conflicts_over_social_media": 3
}
```

### Example Response

```json
{
  "addiction_score": 7.8,
  "addiction_score_pct": 86.7,
  "risk_level": "High",
  "productivity_level": "Low",
  "productivity_probabilities": {"High": 5.2, "Medium": 18.1, "Low": 76.7},
  "score_vs_avg": 1.36,
  "recommendations": [...]
}
```

---

## 🛠️ Tech Stack

| Layer         | Technology                                       |
|---------------|---------------------------------------------------|
| ML Models     | Python, Scikit-learn (Gradient Boosting, Random Forest) |
| API Backend   | Flask, Flask-CORS                                 |
| Frontend      | HTML5, CSS3, Vanilla JavaScript                   |
| Charts        | Chart.js 4.4                                       |
| Fonts         | Syne + DM Sans (Google Fonts)                     |
| BI Dashboard  | Microsoft Power BI                                |
| Dataset       | CSV (705 students)                                |

---

## 📝 Notes

- The frontend uses **graceful fallback** — if the Flask API is not running, it uses a simplified rule-based engine so the UI still works for demonstration purposes.
- All models are saved as `.pkl` files using `joblib` for fast loading.
- CORS is enabled on the Flask server so the frontend can call it from the browser.
