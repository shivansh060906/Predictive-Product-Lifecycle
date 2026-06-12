import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader        import load_m5_data, melt_sales, merge_calendar, select_products
from src.preprocess         import aggregate_sales, handle_missing, get_product_series
from src.feature_engineering import build_feature_matrix, compute_growth_rate, compute_trend_slope
from src.arima_model        import fit_arima, forecast, get_forecast_trend
from src.clustering         import cluster_products, label_clusters
from src.lifecycle          import run_lifecycle_analysis
from src.utils              import plot_forecast, format_results_table

st.set_page_config(page_title='Product Lifecycle Dashboard', layout='wide')
st.title('🛍️ Predictive Product Lifecycle Analysis')

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header('Data Settings')
sales_path    = st.sidebar.text_input('Sales CSV path',    'data/raw/sales_train_evaluation.csv')
calendar_path = st.sidebar.text_input('Calendar CSV path', 'data/raw/calendar.csv')
prices_path   = st.sidebar.text_input('Prices CSV path',   'data/raw/sell_prices.csv')
n_products    = st.sidebar.slider('Number of products to analyse', 5, 50, 20)
store_filter  = st.sidebar.text_input('Filter by store (optional)', '')
forecast_days = st.sidebar.slider('Forecast horizon (days)', 15, 60, 30)

if st.sidebar.button('Load & Run Analysis'):

    with st.spinner('Loading data…'):
        sales, calendar, prices = load_m5_data(sales_path, calendar_path, prices_path)
        sales_long = melt_sales(sales)
        sales_long = merge_calendar(sales_long, calendar)
        subset     = select_products(sales_long, n=n_products, store_id=store_filter or None)

    with st.spinner('Preprocessing…'):
        agg = aggregate_sales(subset)
        agg = handle_missing(agg)

    with st.spinner('Building features…'):
        feature_matrix = build_feature_matrix(agg)
        feature_matrix, kmeans, scaler = cluster_products(feature_matrix)
        feature_matrix = label_clusters(feature_matrix)

    all_results = []
    product_ids = agg['item_id'].unique().tolist()

    progress = st.progress(0)
    for i, item_id in enumerate(product_ids):
        series      = get_product_series(agg, item_id)
        model       = fit_arima(series)
        preds, ci   = forecast(model, steps=forecast_days)
        trend       = get_forecast_trend(preds)
        growth_rate = compute_growth_rate(series)
        variance    = float(series.var())
        result      = run_lifecycle_analysis(
            item_id, growth_rate, trend, variance, preds, float(series.mean())
        )
        result['series'] = series
        result['preds']  = preds
        result['ci']     = ci
        all_results.append(result)
        progress.progress((i + 1) / len(product_ids))

    st.session_state['results']        = all_results
    st.session_state['feature_matrix'] = feature_matrix

# ── Results ───────────────────────────────────────────────────────────────────
if 'results' in st.session_state:
    results        = st.session_state['results']
    feature_matrix = st.session_state['feature_matrix']

    st.subheader('📊 Lifecycle Summary Table')
    summary_df = format_results_table([
        {k: v for k, v in r.items() if k not in ('series', 'preds', 'ci')}
        for r in results
    ])
    st.dataframe(summary_df, use_container_width=True)

    st.subheader('🔍 Individual Product Deep-Dive')
    product_ids = [r['item_id'] for r in results]
    selected    = st.selectbox('Select a product', product_ids)

    if selected:
        r = next(x for x in results if x['item_id'] == selected)

        col1, col2, col3 = st.columns(3)
        col1.metric('Lifecycle Stage', r['lifecycle_stage'])
        col2.metric('Forecast Trend',  r['forecast_trend'])
        col3.metric('Days to Decline', str(r['days_to_decline']))

        st.info(f"💡 **Recommendation:** {r['recommendation']}")

        fig = plot_forecast(r['series'], r['preds'], r['ci'], selected)
        st.pyplot(fig)

    st.subheader('🗂️ Cluster Overview')
    st.dataframe(
        feature_matrix[['item_id', 'avg_sales', 'growth_rate', 'trend_slope', 'cluster_label']],
        use_container_width=True
    )