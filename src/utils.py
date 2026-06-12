import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def setup_logger(name: str = 'lifecycle') -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s — %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_forecast(historical: pd.Series, preds, conf_int, item_id: str, save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(historical.index, historical.values, label='Historical', color='steelblue')

    forecast_index = range(len(historical), len(historical) + len(preds))
    ax.plot(forecast_index, preds, label='Forecast', color='darkorange', linestyle='--')

    ax.fill_between(
        forecast_index,
        conf_int[:, 0],
        conf_int[:, 1],
        alpha=0.2,
        color='darkorange',
        label='95% CI'
    )

    ax.set_title(f'Sales Forecast — {item_id}')
    ax.set_xlabel('Days')
    ax.set_ylabel('Sales')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        return fig


def format_results_table(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df = df.rename(columns={
        'item_id':         'Product',
        'lifecycle_stage': 'Stage',
        'forecast_trend':  'Trend',
        'days_to_decline': 'Days to Decline',
        'recommendation':  'Recommendation'
    })
    return df