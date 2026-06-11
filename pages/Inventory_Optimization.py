import streamlit as st
import pandas as pd
import plotly.express as px
from models import inventory
from database import db_utils
import utils


def app(conn):
    st.header("Inventory Optimization")

    # Load sales from session if available
    df_sales = st.session_state.get('clean_data')

    # Try to load products from DB
    df_products = None
    try:
        prods = db_utils.get_products(conn)
        if not prods.empty:
            df_products = prods.copy()
    except Exception:
        prods = None

    # If sales provide product info, prefer that
    if df_sales is not None and all(c in df_sales.columns for c in ['product_name', 'stock', 'price']):
        df_products = df_sales[['product_id', 'product_name', 'stock', 'price']].drop_duplicates().reset_index(drop=True)

    if df_products is None or df_products.empty:
        st.info("Products data not found in DB or dataset. Provide `product_name`, `stock`, and `price` columns or save products to DB from Data Upload.")
        return

    # Ensure product_id present
    if 'product_id' not in df_products.columns:
        df_products['product_id'] = df_products.index + 1

    # Ensure sales have product_id by mapping product_name when possible
    df_sales_for_calc = pd.DataFrame()
    if df_sales is not None and not df_sales.empty:
        df_sales_for_calc = df_sales.copy()
        if 'product_id' not in df_sales_for_calc.columns and 'product_name' in df_sales_for_calc.columns:
            # build mapping
            try:
                mapping = dict(zip(df_products['product_name'], df_products['product_id']))
                df_sales_for_calc['product_id'] = df_sales_for_calc['product_name'].map(mapping)
            except Exception:
                pass

    # Compute metrics
    try:
        res = inventory.compute_inventory_metrics(df_products.copy(), df_sales_for_calc)
    except Exception as e:
        st.error(f"Inventory computation failed: {e}")
        return

    st.subheader("Inventory Summary")
    res_disp = res.copy()
    if 'price' in res_disp.columns:
        res_disp['price'] = res_disp['price'].apply(utils.format_currency)
    if 'revenue' in res_disp.columns:
        res_disp['revenue'] = res_disp['revenue'].apply(utils.format_currency)
    st.dataframe(res_disp)

    st.subheader("Optimization Recommendations")
    recs = res[['product_id', 'product_name', 'eoq', 'reorder_point', 'stock_coverage_days']]
    st.dataframe(recs)

    # Alerts
    st.subheader("Alerts")
    low_stock = res[res['stock_coverage_days'] < 7]
    overstock = res[res['stock_coverage_days'] > 90]
    if not low_stock.empty:
        st.warning(f"Low stock alert for: {', '.join(low_stock['product_name'].tolist())}")
        st.dataframe(low_stock)
    else:
        st.success("No low stock alerts")

    if not overstock.empty:
        st.info(f"Possible overstock for: {', '.join(overstock['product_name'].tolist())}")
        st.dataframe(overstock)

    # Charts
    st.subheader("Stock Coverage Chart")
    fig = px.bar(res, x='product_name', y='stock_coverage_days', title='Stock Coverage Days per Product')
    st.plotly_chart(fig, use_container_width=True)

    st.download_button('Download Inventory Recommendations CSV', data=res.to_csv(index=False).encode('utf-8'), file_name='inventory_recommendations.csv')

    if st.button('Save Recommendations to DB'):
        with st.spinner('Writing inventory recommendations to DB...'):
            saved = 0
            for _, r in res.iterrows():
                try:
                    pid = int(r.get('product_id')) if not pd.isna(r.get('product_id')) else None
                    stock_level = int(r.get('stock')) if r.get('stock') is not None and not pd.isna(r.get('stock')) else 0
                    rop = int(r.get('reorder_point')) if r.get('reorder_point') is not None and not pd.isna(r.get('reorder_point')) else 0
                    ss = int(r.get('safety_stock')) if r.get('safety_stock') is not None and not pd.isna(r.get('safety_stock')) else 0
                    db_utils.upsert_inventory(conn, pid, stock_level, rop, ss)
                    saved += 1
                except Exception:
                    continue
        st.success(f"Saved {saved} inventory recommendation rows to DB")
