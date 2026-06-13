import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_loader import load_m5_data, melt_sales, merge_calendar, select_products
from src.preprocess import aggregate_sales, handle_missing, get_product_series
from src.prophet_model import fit_prophet, forecast, get_forecast_trend
from src.utils import setup_logger

logger = setup_logger('benchmark_prophet')

HOLDOUT_DAYS = 30


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if mask.sum() == 0:
        return float('inf')
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def naive_forecast(series: pd.Series, steps: int) -> np.ndarray:
    """Last-value-carried-forward baseline."""
    return np.full(steps, series.iloc[-1])


def seasonal_naive_forecast(series: pd.Series, steps: int, period: int = 7) -> np.ndarray:
    """Repeat last week as baseline — better for weekly seasonal retail data."""
    tail = series.iloc[-period:].values
    reps = int(np.ceil(steps / period))
    return np.tile(tail, reps)[:steps]


def evaluate_product(series: pd.Series, item_id: str) -> dict | None:
    if len(series) <= HOLDOUT_DAYS + 30:
        logger.warning(f'{item_id}: not enough data, skipping')
        return None

    train  = series.iloc[:-HOLDOUT_DAYS]
    actual = series.iloc[-HOLDOUT_DAYS:].values

    # ── Prophet ──────────────────────────────────────────────────────────────
    try:
        model      = fit_prophet(train)
        preds, ci  = forecast(model, steps=HOLDOUT_DAYS)
        trend      = get_forecast_trend(preds)
    except Exception as e:
        logger.error(f'{item_id}: Prophet failed — {e}')
        return None

    # ── Baselines ─────────────────────────────────────────────────────────────
    naive          = naive_forecast(train, HOLDOUT_DAYS)
    seasonal_naive = seasonal_naive_forecast(train, HOLDOUT_DAYS)

    # ── Metrics ───────────────────────────────────────────────────────────────
    prophet_mae  = mean_absolute_error(actual, preds)
    prophet_rmse = np.sqrt(mean_squared_error(actual, preds))
    prophet_mape = mape(actual, preds)

    naive_mae    = mean_absolute_error(actual, naive)
    naive_rmse   = np.sqrt(mean_squared_error(actual, naive))
    naive_mape   = mape(actual, naive)

    snaive_mae   = mean_absolute_error(actual, seasonal_naive)
    snaive_rmse  = np.sqrt(mean_squared_error(actual, seasonal_naive))
    snaive_mape  = mape(actual, seasonal_naive)

    beats_naive   = prophet_mae < naive_mae
    beats_snaive  = prophet_mae < snaive_mae

    logger.info(
        f'{item_id} | Trend: {trend} | '
        f'Prophet MAE: {prophet_mae:.2f} | '
        f'Naive MAE: {naive_mae:.2f} | '
        f'SeasonalNaive MAE: {snaive_mae:.2f}'
    )

    return {
        'item_id':          item_id,
        'forecast_trend':   trend,

        'prophet_mae':      round(prophet_mae,  2),
        'prophet_rmse':     round(prophet_rmse, 2),
        'prophet_mape':     round(prophet_mape, 2),

        'naive_mae':        round(naive_mae,    2),
        'naive_rmse':       round(naive_rmse,   2),
        'naive_mape':       round(naive_mape,   2),

        'snaive_mae':       round(snaive_mae,   2),
        'snaive_rmse':      round(snaive_rmse,  2),
        'snaive_mape':      round(snaive_mape,  2),

        'beats_naive':      beats_naive,
        'beats_snaive':     beats_snaive,
    }


if __name__ == '__main__':
    sales, calendar, prices = load_m5_data(
        '../data/raw/sales_train_validation.csv',
        '../data/raw/calendar.csv',
        '../data/raw/sell_prices.csv'
    )
    sales_long = melt_sales(sales)
    sales_long = merge_calendar(sales_long, calendar)
    subset     = select_products(sales_long, n=100)
    agg        = aggregate_sales(subset)
    agg        = handle_missing(agg)

    results = []
    for item_id in agg['item_id'].unique():
        series = get_product_series(agg, item_id)
        result = evaluate_product(series, item_id)
        if result:
            results.append(result)

    df = pd.DataFrame(results)

    print('\n--- Per Product Results ---')
    print(df[[
        'item_id', 'forecast_trend',
        'prophet_mae', 'naive_mae', 'snaive_mae',
        'beats_naive', 'beats_snaive'
    ]].to_string(index=False))

    print('\n--- Aggregate Metrics ---')
    print(f"Prophet   | MAE: {df['prophet_mae'].mean():.2f}  RMSE: {df['prophet_rmse'].mean():.2f}  MAPE: {df['prophet_mape'].mean():.2f}%")
    print(f"Naive     | MAE: {df['naive_mae'].mean():.2f}  RMSE: {df['naive_rmse'].mean():.2f}  MAPE: {df['naive_mape'].mean():.2f}%")
    print(f"SeasonalN | MAE: {df['snaive_mae'].mean():.2f}  RMSE: {df['snaive_rmse'].mean():.2f}  MAPE: {df['snaive_mape'].mean():.2f}%")

    print(f"\nBeats naive baseline:          {df['beats_naive'].sum()}/{len(df)} products")
    print(f"Beats seasonal naive baseline: {df['beats_snaive'].sum()}/{len(df)} products")

    print('\n--- Trend Distribution ---')
    print(df['forecast_trend'].value_counts().to_string())

    df.to_csv('../benchmarks/prophet_benchmark_results.csv', index=False)
    logger.info('Saved → benchmarks/prophet_benchmark_results.csv')