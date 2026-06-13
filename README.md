# Predictive Product Lifecycle Analysis

Predicts the lifecycle stage of retail products (Growth / Maturity / Decline)
and estimates time to decline using the M5 Walmart dataset.

---

## Stack
- **Forecasting** — Facebook Prophet
- **Clustering** — K-Means via `scikit-learn`
- **Dashboard** — Streamlit

---

## Setup

```bash
pip install -r requirements.txt
```

Place M5 dataset files in `data/raw/`:
- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

Or download automatically via Kaggle API:
```bash
python download_data.py
```

---

## Usage

**1. EDA**
```bash
python notebooks/eda.py
```

**2. Prophet Analysis**
```bash
python notebooks/arima_analysis.py
```

**3. Train & Save Models**
```bash
python models/save_models.py
```

**4. Run Benchmarks**
```bash
python benchmarks/benchmark_prophet.py
python benchmarks/benchmark_clustering.py
python benchmarks/benchmark_lifecycle.py
python benchmarks/benchmark_pipeline.py
```

**5. Run Dashboard**
```bash
streamlit run app/app.py
```

---

## Model Details

**Forecasting — Prophet**
- Multiplicative seasonality with yearly and weekly components
- `changepoint_prior_scale=0.5`, `changepoint_range=0.95` for flexible trend detection
- M5 calendar regressors (SNAP flags) for event-driven demand
- 7-day convolution smoothing on forecast output before trend classification

**Clustering — K-Means**
- Features: average sales, growth rate, variance, trend slope
- StandardScaler normalization
- Optimal k validated via Silhouette Score and elbow curve

**Lifecycle Classification — Rule-based**
- Prophet forecast trend is primary signal
- Variance threshold (median of distribution) filters noisy products
- Historical growth rate used as tiebreaker for Stable forecasts

---

## Benchmarks

| Model | Beats Naive | Beats Seasonal Naive |
|---|---|---|
| Prophet | 18/20 | 17/20 |

| Metric | Score |
|---|---|
| Pipeline Trend Accuracy | 46% (99 products) |
| Lifecycle Classifier | Balanced across Growth / Maturity / Decline |

---

## Data

M5 Forecasting Dataset (Walmart) — FOODS_3 department:
- Dense daily sales with low intermittency
- Multi-year time horizon across 10 stores
- Calendar events and SNAP promotion flags

Download from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

---

## Output

- Lifecycle stage per product (Growth / Maturity / Decline)
- Forecast trend direction (Increasing / Stable / Decreasing)
- Days to decline estimate
- Actionable inventory recommendations

---

## Project Structure

```
predictive-lifecycle/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── eda.py
│   └── arima_analysis.py
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── arima_model.py
│   ├── clustering.py
│   ├── lifecycle.py
│   └── utils.py
├── models/
│   ├── save_models.py
│   ├── arima_models.pkl
│   └── clustering.pkl
├── benchmarks/
│   ├── benchmark_prophet.py
│   ├── benchmark_clustering.py
│   ├── benchmark_lifecycle.py
│   └── benchmark_pipeline.py
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```