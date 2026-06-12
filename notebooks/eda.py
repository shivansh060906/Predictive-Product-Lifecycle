import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_m5_data, melt_sales, merge_calendar, select_products
from src.preprocess import aggregate_sales, handle_missing
from src.feature_engineering import build_feature_matrix

# ── Load ──────────────────────────────────────────────────────────────────────
sales, calendar, prices = load_m5_data(
    'data/raw/sales_train_evaluation.csv',
    'data/raw/calendar.csv',
    'data/raw/sell_prices.csv'
)

sales_long = melt_sales(sales)
sales_long = merge_calendar(sales_long, calendar)
subset     = select_products(sales_long, n=20)
agg        = aggregate_sales(subset)
agg        = handle_missing(agg)

# ── Basic Info ────────────────────────────────────────────────────────────────
print(agg.shape)
print(agg.dtypes)
print(agg.describe())
print(agg.isnull().sum())

# ── Sales Distribution ────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
sns.histplot(agg['sales'], bins=50, kde=True)
plt.title('Sales Distribution')
plt.tight_layout()
plt.savefig('data/processed/sales_distribution.png')
plt.close()

# ── Sales Over Time (per product) ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
for item_id, group in agg.groupby('item_id'):
    ax.plot(group['date'], group['sales'], alpha=0.4, linewidth=0.8)
ax.set_title('Sales Over Time — All Products')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.tight_layout()
plt.savefig('data/processed/sales_over_time.png')
plt.close()

# ── Feature Matrix ────────────────────────────────────────────────────────────
feature_matrix = build_feature_matrix(agg)
print(feature_matrix.describe())

sns.pairplot(feature_matrix.drop(columns='item_id'))
plt.savefig('data/processed/feature_pairplot.png')
plt.close()