import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from src.data_loader import load_m5_data, melt_sales, merge_calendar, select_products
from src.preprocess import aggregate_sales, handle_missing, get_product_series
from src.feature_engineering import compute_growth_rate
from src.prophet_model import fit_prophet, forecast, get_forecast_trend
from src.lifecycle import classify_lifecycle
from src.utils import setup_logger

logger = setup_logger('benchmark_lifecycle')

DECLINE_DROP   = 0.10   # 10% HOBBIES products barely sell, drops are small
GROWTH_RISE    = 0.10   # 10% rise = synthetic "Growth" label
EVAL_WINDOW    = 90     # longer window smooths out noise


def synthetic_label(series: pd.Series, window: int = EVAL_WINDOW) -> str:
    """
    Label a product based on actual sales movement in the last `window` days.
    """
    if len(series) < window + 10:
        return None

    baseline = series.iloc[-(window + 10):-(window)].mean()
    recent   = series.iloc[-window:].mean()

    if baseline == 0:
        return None

    change = (recent - baseline) / baseline

    if change <= -DECLINE_DROP:
        return 'Decline'
    elif change >= GROWTH_RISE:
        return 'Growth'
    else:
        return 'Maturity'


if __name__ == '__main__':
    sales, calendar, prices = load_m5_data(
        '../data/raw/sales_train_validation.csv',
        '../data/raw/calendar.csv',
        '../data/raw/sell_prices.csv'
    )
    sales_long = melt_sales(sales)
    sales_long = merge_calendar(sales_long, calendar)
    subset     = select_products(sales_long, n=100, store_id='CA-3', department='FOODS_3')
    agg        = aggregate_sales(subset)
    agg        = handle_missing(agg)

    y_true, y_pred = [], []

    for item_id in agg['item_id'].unique():
        series = get_product_series(agg, item_id)
        label  = synthetic_label(series)
        if label is None:
            continue

        growth_rate = compute_growth_rate(series)
        model       = fit_prophet(series)
        preds, _    = forecast(model, steps=30)
        trend       = get_forecast_trend(preds)
        predicted   = classify_lifecycle(growth_rate, trend, float(series.var()))

        y_true.append(label)
        y_pred.append(predicted)
        logger.info(f'{item_id} | True: {label} | Predicted: {predicted}')

    print('\n--- Classification Report ---')
    print(classification_report(y_true, y_pred, labels=['Growth', 'Maturity', 'Decline']))

    print('--- Confusion Matrix ---')
    cm = confusion_matrix(y_true, y_pred, labels=['Growth', 'Maturity', 'Decline'])
    cm_df = pd.DataFrame(cm,
        index=['True Growth', 'True Maturity', 'True Decline'],
        columns=['Pred Growth', 'Pred Maturity', 'Pred Decline']
    )
    print(cm_df)
    cm_df.to_csv('../benchmarks/lifecycle_confusion_matrix.csv')