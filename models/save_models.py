from src.data_loader import load_m5_data, melt_sales, merge_calendar, select_products
from src.preprocess import aggregate_sales, handle_missing, get_product_series
from src.feature_engineering import build_feature_matrix
from src.arima_model import fit_arima, save_model
from src.clustering import cluster_products, label_clusters, save_clustering
from src.utils import setup_logger, ensure_dir

logger = setup_logger()

# ── Load & Preprocess ─────────────────────────────────────────────────────────
logger.info('Loading data...')
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

ensure_dir('models')

# ── ARIMA — fit one model per product, save as dict ───────────────────────────
logger.info('Fitting ARIMA models...')
arima_models = {}

for item_id in agg['item_id'].unique():
    logger.info(f'  Fitting: {item_id}')
    series = get_product_series(agg, item_id)
    model  = fit_arima(series)
    arima_models[item_id] = model

save_model(arima_models, 'models/arima_models.pkl')
logger.info('Saved → models/arima_models.pkl')

# ── Clustering ────────────────────────────────────────────────────────────────
logger.info('Clustering products...')
feature_matrix = build_feature_matrix(agg)
feature_matrix, kmeans, scaler = cluster_products(feature_matrix)
feature_matrix = label_clusters(feature_matrix)

save_clustering(kmeans, scaler, 'models/clustering.pkl')
logger.info('Saved → models/clustering.pkl')

logger.info('Done.')