import pandas as pd

def aggregate_sales(sales_long: pd.DataFrame) -> pd.DataFrame:
    agg = (
        sales_long
        .groupby(['item_id', 'date'])['sales']
        .sum()  # sum across all 10 stores
        .reset_index()
        .sort_values(['item_id', 'date'])
    )
    return agg


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    filled_frames = []

    for item_id, group in df.groupby('item_id'):
        group = group.set_index('date')
        group = group.asfreq('D', fill_value=0)
        group['item_id'] = item_id
        group = group.reset_index()
        filled_frames.append(group)

    return pd.concat(filled_frames, ignore_index=True)


def get_product_series(df: pd.DataFrame, item_id: str) -> pd.Series:
    product_df = df[df['item_id'] == item_id].copy()
    product_df = product_df.set_index('date').sort_index()
    return product_df['sales']