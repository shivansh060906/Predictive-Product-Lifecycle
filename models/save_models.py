from src.data_loader import load_m5_data, melt_sales, merge_calendar, select_products
from src.preprocess import aggregate_sales, handle_missing, get_product_series
from src.feature_engineering import build_feature_matrix
from src.prophet_model import fit_prophet, save_model
from src.clustering import cluster_products, label_clusters, save_clustering
from src.utils import setup_logger, ensure_dir

logger = setup_logger()

logger.info('Loading data...')
sales, calendar, prices = load_m5_data(
    '../data/raw/sales_train_validation.csv',
    '../data/raw/calendar.csv',
    '../data/raw/sell_prices.csv'
)

sales_long = melt_sales(sales)
sales_long = merge_calendar(sales_long, calendar)
subset = select_products(sales_long, n=100, store_id='CA_3', department='FOODS_3')
agg        = aggregate_sales(subset)
agg        = handle_missing(agg)

ensure_dir('models')

# ── Prophet — one model per product ──────────────────────────────────────────
logger.info('Fitting Prophet models...')
prophet_models = {}

for item_id in agg['item_id'].unique():
    logger.info(f'  Fitting: {item_id}')
    series = get_product_series(agg, item_id)
    model  = fit_prophet(series)
    prophet_models[item_id] = model

save_model(prophet_models, '../models/prophet_models.pkl')  # keeping filename for compatibility
logger.info('Saved → models/prophet_models.pkl')

# ── Clustering ────────────────────────────────────────────────────────────────
logger.info('Clustering products...')
feature_matrix = build_feature_matrix(agg)
feature_matrix, kmeans, scaler = cluster_products(feature_matrix)
feature_matrix = label_clusters(feature_matrix)

save_clustering(kmeans, scaler, '../models/clustering.pkl')
logger.info('Saved → models/clustering.pkl')

logger.info('Done.')