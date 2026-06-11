import streamlit as st
import pandas as pd
import plotly.express as px
import json
from database import db_utils
from utils import df_to_bytes


def app(conn):
    st.header("Business Intelligence Dashboard")
    df = st.session_state.get('clean_data') or st.session_state.get('raw_data')
    if df is None:
        st.info("Upload data to view dashboard")
        return
    st.sidebar.subheader("Filters")
    date_col = 'sale_date' if 'sale_date' in df.columns else None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()

    # load saved views
    saved_views = db_utils.get_saved_views(conn)
    sv_options = ['(none)'] + saved_views['name'].tolist() if not saved_views.empty else ['(none)']
    chosen_view = st.sidebar.selectbox('Saved Views', sv_options)
    loaded_config = None
    if chosen_view != '(none)':
        row = saved_views[saved_views['name'] == chosen_view].iloc[0]
        try:
            loaded_config = json.loads(row['config'])
        except Exception:
            loaded_config = None

    # defaults from loaded view if present
    default_dr = [min_date, max_date]
    default_product = 'All'
    default_category = 'All'
    default_region = 'All'
    if loaded_config:
        try:
            if 'dr' in loaded_config:
                default_dr = [pd.to_datetime(loaded_config['dr'][0]).date(), pd.to_datetime(loaded_config['dr'][1]).date()]
            default_product = loaded_config.get('product', 'All')
            default_category = loaded_config.get('category', 'All')
            default_region = loaded_config.get('region', 'All')
        except Exception:
            pass

    if date_col:
        dr = st.sidebar.date_input("Date range", value=default_dr)
    else:
        dr = None

    product = st.sidebar.selectbox("Product", options=['All'] + (df['product_name'].unique().tolist() if 'product_name' in df.columns else []), index=(0 if default_product=='All' else None))
    category = st.sidebar.selectbox("Category", options=['All'] + (df['category'].unique().tolist() if 'category' in df.columns else []), index=(0 if default_category=='All' else None))
    region = st.sidebar.selectbox("Region", options=['All'] + (df['region'].unique().tolist() if 'region' in df.columns else []), index=(0 if default_region=='All' else None))

    # Save/Delete view controls
    st.sidebar.markdown('---')
    st.sidebar.subheader('Manage Views')
    new_view_name = st.sidebar.text_input('View name')
    if st.sidebar.button('Save current view'):
        cfg = {'dr': [str(dr[0]), str(dr[1])] if dr else None, 'product': product, 'category': category, 'region': region}
        username = st.session_state.get('user', {}).get('username') if st.session_state.get('user') else 'anonymous'
        try:
            db_utils.insert_saved_view(conn, new_view_name or f"view_{pd.Timestamp.now().isoformat()}", username, json.dumps(cfg))
            st.sidebar.success('View saved')
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f'Failed to save view: {e}')

    if chosen_view != '(none)':
        if st.sidebar.button('Delete selected view'):
            # delete by name
            try:
                vid = int(saved_views[saved_views['name'] == chosen_view].iloc[0]['view_id'])
                db_utils.delete_saved_view(conn, vid)
                st.sidebar.success('View deleted')
                st.experimental_rerun()
            except Exception as e:
                st.sidebar.error(f'Failed to delete: {e}')

    dff = df.copy()
    if date_col and dr and len(dr) == 2:
        dff = dff[(dff[date_col].dt.date >= dr[0]) & (dff[date_col].dt.date <= dr[1])]
    if product != 'All' and 'product_name' in dff.columns:
        dff = dff[dff['product_name'] == product]
    if category != 'All' and 'category' in dff.columns:
        dff = dff[dff['category'] == category]
    if region != 'All' and 'region' in dff.columns:
        dff = dff[dff['region'] == region]

    # KPI Cards
    total_revenue = float(dff['revenue'].sum()) if 'revenue' in dff.columns else 0.0
    total_sales = int(dff['quantity'].sum()) if 'quantity' in dff.columns else dff.shape[0]
    total_orders = dff.shape[0]
    inventory_value = 0
    if 'stock' in dff.columns and 'price' in dff.columns:
        inventory_value = (dff['stock'] * dff['price']).sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", f"${total_revenue:,.2f}")
    k2.metric("Total Sales", f"{total_sales:,}")
    k3.metric("Total Orders", f"{total_orders:,}")
    k4.metric("Inventory Value", f"${inventory_value:,.2f}")

    st.subheader("Revenue Trend")
    if 'revenue' in dff.columns and date_col:
        rev = dff.groupby(dff[date_col].dt.to_period('M')).agg({'revenue':'sum'}).reset_index()
        rev['sale_date'] = rev[date_col].dt.to_timestamp()
        fig = px.line(rev, x='sale_date', y='revenue')
        st.plotly_chart(fig, use_container_width=True)
        # allow exporting the revenue trend chart as PNG
        try:
            img_bytes = fig.to_image(format='png')
            st.download_button('Download Revenue Chart (PNG)', data=img_bytes, file_name='revenue_trend.png', mime='image/png')
        except Exception:
            st.info('PNG export requires `kaleido` package. Install `pip install kaleido` to enable chart image downloads.')
    # Export filtered data
    st.markdown('---')
    st.subheader('Export Data')
    csv = df_to_bytes(dff, fmt='csv')
    xlsx = df_to_bytes(dff, fmt='xlsx')
    st.download_button('Download Filtered CSV', data=csv, file_name='filtered_data.csv', mime='text/csv')
    st.download_button('Download Filtered Excel', data=xlsx, file_name='filtered_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    st.subheader("Top Products")
    if 'product_name' in dff.columns and 'quantity' in dff.columns:
        top = dff.groupby('product_name').agg({'quantity':'sum'}).reset_index().sort_values('quantity', ascending=False).head(10)
        fig2 = px.bar(top, x='product_name', y='quantity')
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(top)
        try:
            img2 = fig2.to_image(format='png')
            st.download_button('Download Top Products Chart (PNG)', data=img2, file_name='top_products.png', mime='image/png')
        except Exception:
            pass
