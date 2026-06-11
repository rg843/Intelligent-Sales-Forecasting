import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import utils


def app(conn):
    st.header("Home — Intelligent Sales Forecasting")
    st.subheader("Project Overview")
    st.write("An AI-powered Business Intelligence system for forecasting sales and optimizing inventory.")

    cols = st.columns(4)
    # KPIs from session or placeholder
    df = st.session_state.get('clean_data') or st.session_state.get('raw_data')
    total_revenue = float(df['revenue'].sum()) if df is not None and 'revenue' in df.columns else 0.0
    total_sales = int(df['quantity'].sum()) if df is not None and 'quantity' in df.columns else 0
    total_orders = int(df.shape[0]) if df is not None else 0
    inventory_value = float((df['stock'] * df['price']).sum()) if df is not None and 'stock' in df.columns and 'price' in df.columns else 0.0

    cols[0].metric("Total Revenue", utils.format_currency(total_revenue))
    cols[1].metric("Total Sales", f"{total_sales:,}")
    cols[2].metric("Total Orders", f"{total_orders:,}")
    cols[3].metric("Inventory Value", utils.format_currency(inventory_value))

    st.markdown("---")
    st.subheader("Key Features")
    st.write("- Upload datasets, preprocess data, explore EDA, train models, forecast demand, optimize inventory, and export reports.")

    st.subheader("Revenue Trend (last 180 days)")
    daily = st.session_state.get('daily_agg')
    if daily is not None and not daily.empty:
        import plotly.express as px
        daily['ds'] = pd.to_datetime(daily['ds'])
        fig = px.line(daily.sort_values('ds').tail(180), x='ds', y='y', title='Revenue (daily aggregated)')
        st.plotly_chart(fig, use_container_width=True)
        try:
            img = fig.to_image(format='png')
            st.download_button('Download Revenue Trend PNG', data=img, file_name='revenue_trend.png', mime='image/png')
        except Exception:
            st.info('Install `kaleido` to enable PNG exports for charts.')

    st.subheader("Top Products")
    if df is not None and 'product_name' in df.columns:
        top = df.groupby('product_name').agg({'quantity':'sum'}).reset_index().sort_values('quantity', ascending=False).head(10)
        st.dataframe(top)
