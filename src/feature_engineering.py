import pandas as pd
import numpy as np
from scipy.stats import linregress

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean and variance features per product.
    """
    result_frames = []

    for item_id, group in df.groupby('item_id'):
        group = group.sort_values('date').copy()
        group['rolling_mean_7']  = group['sales'].rolling(7,  min_periods=1).mean()
        group['rolling_mean_30'] = group['sales'].rolling(30, min_periods=1).mean()
        group['rolling_var_7']   = group['sales'].rolling(7,  min_periods=1).var().fillna(0)
        result_frames.append(group)

    return pd.concat(result_frames, ignore_index=True)


def compute_growth_rate(series: pd.Series, window: int = 30) -> float:
    """
    Compute percentage growth rate over a trailing window.
    """
    if len(series) < window:
        return 0.0
    recent = series.iloc[-window:]
    if recent.iloc[0] == 0:
        return 0.0
    return float((recent.iloc[-1] - recent.iloc[0]) / (recent.iloc[0] + 1e-6))


def compute_trend_slope(series: pd.Series) -> float:
    """
    Fit a linear regression and return the slope as trend indicator.
    """
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    slope, _, _, _, _ = linregress(x, series.values)
    return float(slope)


def compute_variance(series: pd.Series) -> float:
    return float(series.var())


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-product feature matrix for clustering.
    """
    records = []
    for item_id, group in df.groupby('item_id'):
        series = group.sort_values('date')['sales']
        records.append({
            'item_id':     item_id,
            'avg_sales':   series.mean(),
            'growth_rate': compute_growth_rate(series),
            'variance':    compute_variance(series),
            'trend_slope': compute_trend_slope(series),
        })
    return pd.DataFrame(records)