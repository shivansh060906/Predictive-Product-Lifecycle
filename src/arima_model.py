import pandas as pd
import numpy as np
import pickle
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm


def check_stationarity(series: pd.Series) -> dict:
    result = adfuller(series.dropna())
    return {
        'adf_statistic': result[0],
        'p_value':       result[1],
        'is_stationary': result[1] < 0.05
    }


def fit_arima(series: pd.Series, seasonal: bool = False) -> pm.ARIMA:
    model = pm.auto_arima(
        series,
        seasonal=seasonal,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        max_p=3, max_q=3, max_d=2
    )
    return model


def forecast(model: pm.ARIMA, steps: int = 30) -> np.ndarray:
    preds, conf_int = model.predict(n_periods=steps, return_conf_int=True)
    return preds, conf_int


def get_forecast_trend(preds: np.ndarray, threshold: float = 0.05) -> str:
    start = np.mean(preds[:5])
    end   = np.mean(preds[-5:])
    if start == 0:
        return 'Stable'
    change = (end - start) / (start + 1e-6)
    if change > threshold:
        return 'Increasing'
    elif change < -threshold:
        return 'Decreasing'
    else:
        return 'Stable'


def save_model(model, path: str):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)