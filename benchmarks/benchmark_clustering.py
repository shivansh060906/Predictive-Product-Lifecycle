import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.data_loader import load_m5_data, melt_sales, merge_calendar, select_products
from src.preprocess import aggregate_sales, handle_missing
from src.feature_engineering import build_feature_matrix
from src.utils import setup_logger

logger = setup_logger('benchmark_clustering')

FEATURES = ['avg_sales', 'growth_rate', 'variance', 'trend_slope']


def elbow_curve(X_scaled, max_k: int = 8):
    inertias = []
    k_range  = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), inertias, marker='o')
    plt.title('Elbow Curve — KMeans Inertia vs k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (WCSS)')
    plt.tight_layout()
    plt.savefig('../benchmarks/elbow_curve.png')
    plt.close()
    logger.info('Elbow curve saved → benchmarks/elbow_curve.png')
    return dict(zip(k_range, inertias))


def silhouette_scores(X_scaled, max_k: int = 8):
    scores  = {}
    k_range = range(2, max_k + 1)

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        scores[k] = score
        logger.info(f'k={k} | Silhouette Score: {score:.4f}')

    best_k = max(scores, key=scores.get)
    print(f'\nBest k by Silhouette Score: {best_k} (score={scores[best_k]:.4f})')
    return scores


if __name__ == '__main__':
    sales, calendar, prices = load_m5_data(
        '../data/raw/sales_train_validation.csv',
        '../data/raw/calendar.csv',
        '../data/raw/sell_prices.csv'
    )
    sales_long = melt_sales(sales)
    sales_long = merge_calendar(sales_long, calendar)
    subset     = select_products(sales_long, n=20)
    agg        = aggregate_sales(subset)
    agg        = handle_missing(agg)

    feature_matrix = build_feature_matrix(agg)
    X = feature_matrix[FEATURES].fillna(0)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print('--- Elbow Curve ---')
    elbow_curve(X_scaled)

    print('\n--- Silhouette Scores ---')
    silhouette_scores(X_scaled)