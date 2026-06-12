# Predictive Product Lifecycle Analysis

Predicts the lifecycle stage of retail products (Growth / Maturity / Decline)
and estimates time to decline using the M5 Walmart dataset.

---

## Stack
- **Forecasting** — ARIMA via `pmdarima`
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

---

## Usage

**1. EDA**
```bash
python notebooks/eda.py
```

**2. ARIMA Analysis**
```bash
python notebooks/arima_analysis.py
```

**3. Train & Save Models**
```bash
python models/save_models.py
```

**4. Run Dashboard**
```bash
streamlit run app/app.py
```

---

## Output

- Lifecycle stage per product
- Forecast trend (Increasing / Stable / Decreasing)
- Days to decline estimate
- Actionable recommendations

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
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```