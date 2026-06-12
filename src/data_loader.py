import pandas as pd
import os

def load_m5_data(sales_path: str, calendar_path: str, prices_path: str):
    """
    Load M5 dataset files.
    """
    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path)
    return sales, calendar, prices


def melt_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Melt wide-format sales data into long format.
    """
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    day_cols = [c for c in sales.columns if c.startswith('d_')]

    sales_long = sales.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name='d',
        value_name='sales'
    )
    return sales_long


def merge_calendar(sales_long: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Merge calendar info to get actual dates.
    """
    sales_long = sales_long.merge(calendar[['d', 'date', 'wday', 'month', 'year']], on='d', how='left')
    sales_long['date'] = pd.to_datetime(sales_long['date'])
    return sales_long


def select_products(sales_long: pd.DataFrame, n: int = 20, store_id: str = None) -> pd.DataFrame:
    """
    Select a subset of products for analysis.
    """
    if store_id:
        sales_long = sales_long[sales_long['store_id'] == store_id]

    top_products = sales_long['item_id'].value_counts().head(n).index.tolist()
    return sales_long[sales_long['item_id'].isin(top_products)]