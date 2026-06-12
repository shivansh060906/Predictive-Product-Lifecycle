import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_loader import load_m5_data, melt_sales, merge_calendar, select_products
from src.preprocess import aggregate_sales, handle_missing, get_product_series
from src.arima_model import fit_arima, forecast
from src.utils import setup_logger

logger = setup_logger('benchmark_arima')

HOLDOUT_DAYS = 30


def naive_forecast(series: pd.Series, steps: int) -> np.ndarray:
    """Last-value-carried-forward baseline."""
    return np.full(steps, series.iloc[-1])


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def evaluate_product(series: pd.Series, item_id: str) -> dict:
    if len(series) <= HOLDOUT_DAYS + 10:
        return None

    train = series.iloc[:-HOLDOUT_DAYS]
    actual = series.iloc[-HOLDOUT_DAYS:].values

    # ARIMA
    model = fit_arima(train)
    preds, _ = forecast(model, steps=HOLDOUT_DAYS)

    # Naive baseline
    naive = naive_forecast(train, HOLDOUT_DAYS)

    arima_mae  = mean_absolute_error(actual, preds)
    arima_rmse = np.sqrt(mean_squared_error(actual, preds))
    arima_mape = mape(actual, preds)

    naive_mae  = mean_absolute_error(actual, naive)
    naive_rmse = np.sqrt(mean_squared_error(actual, naive))
    naive_mape = mape(actual, naive)

    logger.info(f'{item_id} | ARIMA MAE: {arima_mae:.2f}  Naive MAE: {naive_mae:.2f}')

    return {
        'item_id':    item_id,
        'arima_mae':  arima_mae,
        'arima_rmse': arima_rmse,
        'arima_mape': arima_mape,
        'naive_mae':  naive_mae,
        'naive_rmse': naive_rmse,
        'naive_mape': naive_mape,
        'arima_beats_naive': arima_mae < naive_mae,
    }


if __name__ == '__main__':
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

    results = []
    for item_id in agg['item_id'].unique():
        series = get_product_series(agg, item_id)
        result = evaluate_product(series, item_id)
        if result:
            results.append(result)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print(f"\nARIMA beats naive baseline: {df['arima_beats_naive'].sum()}/{len(df)} products")
    df.to_csv('../benchmarks/arima_benchmark_results.csv', index=False)