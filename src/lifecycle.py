import pandas as pd


GROWTH_THRESHOLD   = 0.10   # 5% growth rate
DECLINE_THRESHOLD  = -0.10  # -5% growth rate
VARIANCE_THRESHOLD = 33.0   # obtained from data median


def classify_lifecycle(
    growth_rate: float,
    forecast_trend: str,
    variance: float
) -> str:
    if forecast_trend == 'Increasing' and variance < VARIANCE_THRESHOLD:
        return 'Growth'
    elif forecast_trend == 'Decreasing' and variance < VARIANCE_THRESHOLD:
        return 'Decline'
    elif forecast_trend in ('Increasing', 'Decreasing') and variance >= VARIANCE_THRESHOLD:
        return 'Maturity'
    else:
        # Stable forecast
        if growth_rate > 0.20 and variance < 100:
            return 'Growth'
        elif growth_rate < -0.15 and variance < 100:
            return 'Decline'
        else:
            return 'Maturity'


def estimate_time_to_decline(
    preds,
    current_avg: float,
    drop_threshold: float = 0.3
) -> int:
    target = current_avg * (1 - drop_threshold)
    for i, val in enumerate(preds):
        if val < target:
            return i
    return -1


def generate_recommendation(stage: str, days_to_decline: int) -> str:
    if stage == 'Growth':
        return 'Product is in growth phase. Scale inventory and increase marketing spend.'
    elif stage == 'Maturity':
        if days_to_decline != -1 and days_to_decline <= 30:
            return f'Product is maturing. Decline expected in ~{days_to_decline} days. Consider promotions or a refresh.'
        return 'Product is stable. Monitor closely and plan for a refresh cycle.'
    else:
        return 'Product is declining. Consider discounting, bundling, or retirement.'


def run_lifecycle_analysis(
    item_id: str,
    growth_rate: float,
    forecast_trend: str,
    variance: float,
    preds,
    current_avg: float
) -> dict:

    stage           = classify_lifecycle(growth_rate, forecast_trend, variance)
    days_to_decline = estimate_time_to_decline(preds, current_avg)
    recommendation  = generate_recommendation(stage, days_to_decline)

    return {
        'item_id':         item_id,
        'lifecycle_stage': stage,
        'forecast_trend':  forecast_trend,
        'days_to_decline': days_to_decline if days_to_decline != -1 else 'N/A',
        'recommendation':  recommendation,
    }