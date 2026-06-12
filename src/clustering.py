import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_products(
    feature_matrix: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42
) -> pd.DataFrame:
    features = ['avg_sales', 'growth_rate', 'variance', 'trend_slope']
    X = feature_matrix[features].fillna(0)

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    feature_matrix = feature_matrix.copy()
    feature_matrix['cluster'] = kmeans.fit_predict(X_scaled)

    return feature_matrix, kmeans, scaler


def label_clusters(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    cluster_summary = (
        feature_matrix
        .groupby('cluster')['growth_rate']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    label_map = {}
    labels = ['Growing', 'Stable', 'Declining']
    for i, row in cluster_summary.iterrows():
        label_map[row['cluster']] = labels[i]

    feature_matrix = feature_matrix.copy()
    feature_matrix['cluster_label'] = feature_matrix['cluster'].map(label_map)
    return feature_matrix


def save_clustering(kmeans, scaler, path: str):
    with open(path, 'wb') as f:
        pickle.dump({'kmeans': kmeans, 'scaler': scaler}, f)


def load_clustering(path: str):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj['kmeans'], obj['scaler']