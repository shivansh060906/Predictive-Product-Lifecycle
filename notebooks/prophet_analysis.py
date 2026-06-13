import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.data_loader import load_m5_data, melt_sales, merge_calendar, select_products
from src.preprocess import aggregate_sales, handle_missing, get_product_series
from src.prophet_model import check_stationarity, fit_prophet, forecast, get_forecast_trend
from src.utils import plot_forecast

# ── Load ──────────────────────────────────────────────────────────────────────
sales, calendar, prices = load_m5_data(
    '../data/raw/sales_train_validation.csv',
    '../data/raw/calendar.csv',
    '../data/raw/sell_prices.csv'
)

sales_long = melt_sales(sales)
sales_long = merge_calendar(sales_long, calendar)
subset     = select_products(sales_long, n=20)
agg        = aggregate_sales(subset)
agg        = handle_missing(agg)

# ── Pick one product for analysis ─────────────────────────────────────────────
ITEM_ID = agg['item_id'].unique()[0]
series  = get_product_series(agg, ITEM_ID)
print(f'Analysing: {ITEM_ID} | Length: {len(series)}')

# ── Stationarity ──────────────────────────────────────────────────────────────
result = check_stationarity(series)
print(f"ADF p-value: {result['p_value']:.4f} — Stationary: {result['is_stationary']}")

# ── Fit & Forecast ────────────────────────────────────────────────────────────
model = fit_prophet(series)
preds, ci = forecast(model, steps=30)
trend = get_forecast_trend(preds)

print(f'Order: {model.order}')
print(f'Forecast trend: {trend}')
print(model.summary())

fig = plot_forecast(series, preds, ci, ITEM_ID)
fig.savefig('../data/processed/forecast.png')
plt.close()