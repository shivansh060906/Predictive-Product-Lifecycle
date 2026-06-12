import pandas as pd

def aggregate_sales(sales_long: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sales per product across all stores.
    """
    agg = (
        sales_long
        .groupby(['item_id', 'date'])['sales']
        .sum()
        .reset_index()
        .sort_values(['item_id', 'date'])
    )
    return agg


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing dates and zero-fill sales gaps.
    """
    filled_frames = []

    for item_id, group in df.groupby('item_id'):
        group = group.set_index('date')
        group = group.asfreq('D', fill_value=0)
        group['item_id'] = item_id
        group = group.reset_index()
        filled_frames.append(group)

    return pd.concat(filled_frames, ignore_index=True)


def get_product_series(df: pd.DataFrame, item_id: str) -> pd.Series:
    """
    Return a clean time series for a single product.
    """
    product_df = df[df['item_id'] == item_id].copy()
    product_df = product_df.set_index('date').sort_index()
    return product_df['sales']