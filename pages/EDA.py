import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from database import db_utils
import utils


def app(conn):
    st.header("Exploratory Data Analysis")
    # prefer cleaned session data, then raw session data, then DB sales
    df = st.session_state.get('clean_data') or st.session_state.get('raw_data')
    if df is None:
        try:
            # attempt to load from DB
            sales = db_utils.get_sales(conn)
            products = db_utils.get_products(conn)
            if not sales.empty:
                df = sales.merge(products, on='product_id', how='left')
                st.session_state['raw_data'] = df
        except Exception:
            df = None

    if df is None:
        st.info("Upload a dataset first")
        return

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include='all').transpose())
    # allow downloads of summary
    csv_summary = df.describe(include='all').transpose().to_csv().encode('utf-8')
    st.download_button("Download Summary CSV", data=csv_summary, file_name='eda_summary.csv', mime='text/csv')

    # Sales Analysis
    st.subheader("Sales Analysis")
    if 'sale_date' in df.columns and 'quantity' in df.columns:
        df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
        df['date'] = df['sale_date'].dt.date

        daily = df.groupby('date').agg({'quantity': 'sum', 'revenue': 'sum' if 'revenue' in df.columns else 'sum'}).reset_index()
        fig_daily = px.line(daily, x='date', y='quantity', title='Daily Sales')
        st.plotly_chart(fig_daily, use_container_width=True)
        daily_disp = daily.copy()
        if 'revenue' in daily_disp.columns:
            daily_disp['revenue'] = daily_disp['revenue'].apply(utils.format_currency)
        st.dataframe(daily_disp.tail(20))
        st.download_button('Download Daily Aggregates CSV', data=daily.to_csv(index=False).encode('utf-8'), file_name='daily_aggregates.csv')
        try:
            img = fig_daily.to_image(format='png')
            st.download_button('Download Daily Sales PNG', data=img, file_name='daily_sales.png', mime='image/png')
        except Exception:
            st.info('Install `kaleido` to enable PNG exports for charts.')

        weekly = df.groupby(pd.Grouper(key='sale_date', freq='W')).agg({'quantity': 'sum'}).reset_index()
        fig_weekly = px.line(weekly, x='sale_date', y='quantity', title='Weekly Sales')
        st.plotly_chart(fig_weekly, use_container_width=True)
        try:
            img = fig_weekly.to_image(format='png')
            st.download_button('Download Weekly Sales PNG', data=img, file_name='weekly_sales.png', mime='image/png')
        except Exception:
            st.info('Install `kaleido` to enable PNG exports for charts.')

        monthly = df.groupby(pd.Grouper(key='sale_date', freq='M')).agg({'quantity': 'sum'}).reset_index()
        fig_monthly = px.line(monthly, x='sale_date', y='quantity', title='Monthly Sales')
        st.plotly_chart(fig_monthly, use_container_width=True)
        try:
            img = fig_monthly.to_image(format='png')
            st.download_button('Download Monthly Sales PNG', data=img, file_name='monthly_sales.png', mime='image/png')
        except Exception:
            st.info('Install `kaleido` to enable PNG exports for charts.')

    # Product Analysis
    st.subheader("Product Analysis")
    if 'product_name' in df.columns and 'quantity' in df.columns:
        prod = df.groupby('product_name').agg({'quantity': 'sum'}).reset_index().sort_values('quantity', ascending=False)
        top = prod.head(10)
        bottom = prod.tail(10)
        fig_top = px.bar(top, x='product_name', y='quantity', title='Top Selling Products')
        st.plotly_chart(fig_top, use_container_width=True)
        st.dataframe(top)
        st.download_button('Download Product Summary CSV', data=prod.to_csv(index=False).encode('utf-8'), file_name='product_summary.csv')
        try:
            img = fig_top.to_image(format='png')
            st.download_button('Download Top Products PNG', data=img, file_name='top_products.png', mime='image/png')
        except Exception:
            st.info('Install `kaleido` to enable PNG exports for charts.')

        fig_bot = px.bar(bottom, x='product_name', y='quantity', title='Bottom Selling Products')
        st.plotly_chart(fig_bot, use_container_width=True)
        st.dataframe(bottom)

    # Category Analysis
    if 'category' in df.columns and 'quantity' in df.columns:
        cat = df.groupby('category').agg({'quantity': 'sum'}).reset_index()
        fig_cat = px.pie(cat, names='category', values='quantity', title='Product Category Distribution')
        st.plotly_chart(fig_cat, use_container_width=True)
        st.dataframe(cat)
        st.download_button('Download Category Summary CSV', data=cat.to_csv(index=False).encode('utf-8'), file_name='category_summary.csv')
        try:
            img = fig_cat.to_image(format='png')
            st.download_button('Download Category Sales PNG', data=img, file_name='category_sales.png', mime='image/png')
        except Exception:
            st.info('Install `kaleido` to enable PNG exports for charts.')

    # Regional Analysis
    st.subheader("Regional Analysis")
    if 'region' in df.columns and 'revenue' in df.columns:
        reg = df.groupby('region').agg({'revenue': 'sum', 'quantity': 'sum'}).reset_index()
        fig_reg = px.bar(reg, x='region', y='revenue', title='Region-wise Revenue')
        st.plotly_chart(fig_reg, use_container_width=True)
        reg_disp = reg.copy()
        if 'revenue' in reg_disp.columns:
            reg_disp['revenue'] = reg_disp['revenue'].apply(utils.format_currency)
        st.dataframe(reg_disp)
        st.download_button('Download Region Summary CSV', data=reg.to_csv(index=False).encode('utf-8'), file_name='region_summary.csv')
        try:
            img = fig_reg.to_image(format='png')
            st.download_button('Download Region Sales PNG', data=img, file_name='region_sales.png', mime='image/png')
        except Exception:
            st.info('Install `kaleido` to enable PNG exports for charts.')

    # Customer Analysis
    st.subheader("Customer Analysis")
    if 'customer_id' in df.columns and 'revenue' in df.columns:
        cust = df.groupby('customer_id').agg({'revenue': 'sum', 'quantity': 'sum'}).reset_index().sort_values('revenue', ascending=False)
        cust_disp = cust.copy()
        if 'revenue' in cust_disp.columns:
            cust_disp['revenue'] = cust_disp['revenue'].apply(utils.format_currency)
        st.dataframe(cust_disp.head(20))

    # Statistical Analysis
    st.subheader("Statistical Analysis")
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] > 1:
        corr = num.corr()
        fig_corr = px.imshow(corr, text_auto=True, title='Correlation Matrix')
        st.plotly_chart(fig_corr, use_container_width=True)
        st.dataframe(corr)
        st.download_button('Download Correlation CSV', data=corr.to_csv().encode('utf-8'), file_name='correlation.csv')
        try:
            img = fig_corr.to_image(format='png')
            st.download_button('Download Correlation Matrix PNG', data=img, file_name='correlation_matrix.png', mime='image/png')
        except Exception:
            st.info('Install `kaleido` to enable PNG exports for charts.')

    # Additional visualizations
    st.subheader("Distributions & Outliers")
    for c in num.columns[:4]:
        fig_hist = px.histogram(df, x=c, nbins=30, title=f'Distribution of {c}')
        fig_box = px.box(df, y=c, title=f'Boxplot of {c}')
        st.plotly_chart(fig_hist, use_container_width=True)
        st.plotly_chart(fig_box, use_container_width=True)
