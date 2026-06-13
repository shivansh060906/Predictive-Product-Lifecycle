import numpy as np
import pandas as pd
import pickle
from prophet import Prophet


def fit_prophet(series: pd.Series, calendar: pd.DataFrame = None) -> Prophet:
    df = pd.DataFrame({'ds': series.index, 'y': series.values})

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.5,
        changepoint_range=0.95,
        seasonality_mode='multiplicative'
    )

    # Add SNAP and event flags if calendar provided
    if calendar is not None:
        snap_cols = [c for c in calendar.columns if c.startswith('snap_')]
        for col in snap_cols:
            cal_sub = calendar[['date', col]].rename(columns={'date': 'ds', col: col})
            cal_sub['ds'] = pd.to_datetime(cal_sub['ds'])
            model.add_regressor(col)
            df = df.merge(cal_sub, on='ds', how='left').fillna(0)

    model.fit(df)
    return model


def forecast(model: Prophet, steps: int = 30):
    future = model.make_future_dataframe(periods=steps)
    forecast_df = model.predict(future)
    preds = forecast_df['yhat'].iloc[-steps:].values
    lower = forecast_df['yhat_lower'].iloc[-steps:].values
    upper = forecast_df['yhat_upper'].iloc[-steps:].values
    conf_int = np.column_stack([lower, upper])
    return preds, conf_int


def get_forecast_trend(preds: np.ndarray, threshold: float = 0.08) -> str:
    # smooth first to reduce seasonal noise
    preds = np.convolve(preds, np.ones(7)/7, mode='valid')
    n = len(preds)
    start = np.mean(preds[:n//3])
    end   = np.mean(preds[-n//3:])
    if start == 0:
        return 'Stable'
    change = (end - start) / (abs(start) + 1e-6)
    if change > threshold:
        return 'Increasing'
    elif change < -threshold:
        return 'Decreasing'
    else:
        return 'Stable'


def check_stationarity(series: pd.Series) -> dict:
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series.dropna())
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] < 0.05
    }


def save_model(model, path: str):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)