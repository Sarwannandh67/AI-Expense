# 🚀 AI Expense Intelligence System

> **End-to-end ML pipeline**: NLP Categorization → Clustering → Anomaly Detection → Forecasting → Streamlit Dashboard

---

## 🏗️ Architecture

```
CSV Input (Synthetic / Your Own)
         ↓
  Data Cleaning & Feature Engineering
         ↓
  ┌──────────────────────────────────┐
  │   Module 1: NLP Categorizer      │  TF-IDF + Logistic Regression
  │   Module 2: Behavior Clustering  │  KMeans + Elbow Method
  │   Module 3: Anomaly Detection    │  Isolation Forest
  │   Module 4: Savings Forecasting  │  Prophet / Linear Trend
  └──────────────────────────────────┘
         ↓
  Streamlit Dashboard (4 Pages)
```

---

## 📁 Project Structure

```
expense_ai/
├── app.py                      # Main Streamlit dashboard
├── requirements.txt
├── data/
│   └── generate_data.py        # Synthetic Indian transaction generator
└── modules/
    ├── categorizer.py          # Module 1: NLP categorization
    ├── clustering.py           # Module 2: KMeans spending profiles
    ├── anomaly_detector.py     # Module 3: Isolation Forest
    └── forecaster.py           # Module 4: Prophet time-series
```

---

## ⚙️ Setup

```bash
# 1. Clone / unzip the project
cd expense_ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

---

## 📦 Modules

### Module 1 — NLP Expense Categorizer
**File**: `modules/categorizer.py`

| Component | Detail |
|-----------|--------|
| Vectorizer | TF-IDF with bigrams, 5000 features |
| Classifier | Logistic Regression (C=5, balanced) |
| Text Preprocessing | Lowercase → Remove numbers → Remove noise words → N-grams |
| Evaluation | Accuracy, F1-score, Confusion Matrix per category |
| Categories | Food, Rent, Transport, Subscriptions, Entertainment, Utilities, Others |

```python
from modules.categorizer import build_categorizer, predict_category
results = build_categorizer(df)
pred = predict_category(results['model'], "Swiggy Order")
# → {'predicted_category': 'Food', 'confidence': 97.2, ...}
```

---

### Module 2 — Spending Behavior Clustering
**File**: `modules/clustering.py`

| Cluster | Profile | Description |
|---------|---------|-------------|
| 0 | 🏦 Conservative Saver | Low spend, high savings |
| 1 | ⚖️ Balanced Spender | Well-distributed across categories |
| 2 | 🛍️ Impulsive Spender | High frequency, Entertainment heavy |
| 3 | 📱 Subscription Heavy | Multiple active subscriptions |

**Features Used**:
- % income spent, savings rate
- Category distribution (normalized)
- Transaction frequency & average value
- Peak transaction amount

```python
from modules.clustering import extract_clustering_features, run_clustering
feature_df = extract_clustering_features(df, income=75000)
results = run_clustering(feature_df)
```

---

### Module 3 — Anomaly Detection
**File**: `modules/anomaly_detector.py`

| Component | Detail |
|-----------|--------|
| Model | Isolation Forest (n_estimators=200) |
| Contamination | 5% (tunable) |
| Features | Amount, log-amount, z-score within category, daily frequency, personal baseline |
| Output | Boolean flag + confidence score + human-readable explanation |

```python
from modules.anomaly_detector import detect_anomalies, get_anomaly_summary
df_flagged, model, scaler = detect_anomalies(df, contamination=0.05)
summary = get_anomaly_summary(df_flagged)
# → [{'date': '2024-04-15', 'description': 'Electronics Store', 
#     'amount': 32000, 'explanation': 'Amount is 4.2x your typical Others spend'}]
```

---

### Module 4 — Savings Forecasting
**File**: `modules/forecaster.py`

| Component | Detail |
|-----------|--------|
| Primary Model | Facebook Prophet (trend + seasonality) |
| Fallback | Linear Regression trend |
| Output | 3-month expense & savings forecast with confidence interval |
| Health Score | 0–100 composite score (savings rate + consistency + subscriptions + anomalies) |

```python
from modules.forecaster import prepare_time_series, forecast_with_prophet, compute_financial_health_score
monthly_df = prepare_time_series(df, monthly_income=75000)
results = forecast_with_prophet(monthly_df)
health = compute_financial_health_score(df, 75000)
# → {'score': 68.5, 'grade': 'B', 'message': 'Good financial habits...'}
```

---

## 📊 Dashboard Pages

| Page | What You See |
|------|-------------|
| 📊 Dashboard | KPIs, monthly trend, category pie, heatmap |
| 🧠 AI Insights | Financial health score, anomaly list, spending profile |
| 🔮 Forecast | 3-month predictions, savings trend, confidence bands |
| 📋 Transactions | Searchable, filterable transaction explorer with CSV export |

---

## 🎯 Bullet Points

```
• Built AI Expense Intelligence System processing 500+ monthly transactions using 
  TF-IDF + Logistic Regression NLP classifier (95%+ accuracy) for automatic 
  expense categorization across 7 categories.

• Implemented Isolation Forest anomaly detection identifying unusual spending 
  patterns (3x-category-average transactions, frequency spikes) with confidence 
  scoring and human-readable explanations.

• Designed KMeans clustering (k=4 via elbow method) to segment spending behavior 
  into Conservative/Balanced/Impulsive/Subscription-Heavy profiles with actionable 
  financial advice.

• Engineered Facebook Prophet time-series model forecasting 3-month expense and 
  savings trends with confidence intervals; deployed end-to-end as Streamlit 
  dashboard with 4 interactive pages.
```

---

## 🔧 Extending the Project

### Add BERT Categorizer (Advanced)
```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
categories = ["Food", "Rent", "Transport", "Subscriptions", "Entertainment", "Utilities", "Others"]
result = classifier("Swiggy Order", candidate_labels=categories)
```

### Add SHAP Explainability
```python
import shap
vectorizer = model_pipeline.named_steps['tfidf']
clf = model_pipeline.named_steps['clf']
explainer = shap.LinearExplainer(clf, vectorizer.transform(X_train))
shap_values = explainer.shap_values(vectorizer.transform(X_test))
```

### Add LSTM Forecasting
```python
# In modules/forecaster.py, replace forecast_with_prophet with LSTM
# Use sliding window of 3 months to predict month 4
# Scale with MinMaxScaler, train PyTorch LSTM
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Pandas, NumPy |
| NLP | Scikit-learn TF-IDF, Logistic Regression |
| Clustering | KMeans, PCA |
| Anomaly | Isolation Forest |
| Forecasting | Prophet, Linear Regression |
| Visualization | Plotly, Streamlit |
| Explainability | SHAP (optional extension) |

---

*Built as a portfolio project demonstrating end-to-end ML pipeline design, multiple ML paradigms, and product-grade deployment.*
