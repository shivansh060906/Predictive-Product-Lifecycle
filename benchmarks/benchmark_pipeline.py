import pandas as pd

from src.data_loader import load_m5_data, melt_sales, merge_calendar, select_products
from src.feature_engineering import compute_growth_rate
from src.lifecycle import run_lifecycle_analysis
from src.preprocess import aggregate_sales, handle_missing, get_product_series
from src.prophet_model import fit_prophet, forecast, get_forecast_trend
from src.utils import setup_logger

logger = setup_logger('benchmark_pipeline')

# Train window: all data except last 60 days
# Eval window:  last 60 days
EVAL_DAYS = 60


if __name__ == '__main__':
    sales, calendar, prices = load_m5_data(
        '../data/raw/sales_train_validation.csv',
        '../data/raw/calendar.csv',
        '../data/raw/sell_prices.csv'
    )
    sales_long = melt_sales(sales)
    sales_long = merge_calendar(sales_long, calendar)
    subset = select_products(sales_long, n=100, store_id='CA_3')
    agg        = aggregate_sales(subset)
    agg        = handle_missing(agg)

    # Split by date
    all_dates  = sorted(agg['date'].unique())
    cutoff     = all_dates[-EVAL_DAYS]
    train_agg  = agg[agg['date'] < cutoff]
    eval_agg   = agg[agg['date'] >= cutoff]

    logger.info(f'Train up to: {cutoff} | Eval from: {cutoff}')

    results = []
    for item_id in train_agg['item_id'].unique():
        train_series = get_product_series(train_agg, item_id)
        eval_series  = get_product_series(eval_agg, item_id)

        if len(train_series) < 30 or len(eval_series) == 0:
            continue

        # Predict on train, check against eval
        growth_rate = compute_growth_rate(train_series)
        model       = fit_prophet(train_series)
        preds, _    = forecast(model, steps=EVAL_DAYS)
        trend       = get_forecast_trend(preds)

        result = run_lifecycle_analysis(
            item_id, growth_rate, trend,
            float(train_series.var()), preds,
            float(train_series.mean())
        )

        # Actual trend in eval window
        actual_change = (eval_series.mean() - train_series.mean()) / (train_series.mean() + 1e-6)

        if abs(actual_change) > 0.90:
            logger.warning(f'{item_id}: extreme change {actual_change:.0%}, likely delisted — skipping')
            continue

        if actual_change > 0.15:
            actual_trend = 'Increasing'
        elif actual_change < -0.15:
            actual_trend = 'Decreasing'
        else:
            actual_trend = 'Stable'

        result['actual_trend']         = actual_trend
        result['forecast_correct']     = result['forecast_trend'] == actual_trend
        result['actual_sales_change%'] = round(actual_change * 100, 2)
        results.append(result)

        logger.info(
            f"{item_id} | Predicted: {result['forecast_trend']} | "
            f"Actual: {actual_trend} | Match: {result['forecast_correct']}"
        )

    df = pd.DataFrame(results)
    accuracy = df['forecast_correct'].mean() * 100
    print(f'\nPipeline Trend Accuracy: {accuracy:.1f}% over {len(df)} products')
    print(df[['item_id', 'lifecycle_stage', 'forecast_trend', 'actual_trend', 'actual_sales_change%']].to_string(index=False))
    df.to_csv('../benchmarks/pipeline_benchmark_results.csv', index=False)