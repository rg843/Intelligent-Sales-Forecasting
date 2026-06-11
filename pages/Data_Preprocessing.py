import streamlit as st
import pandas as pd
import numpy as np
from models import preprocessing
from utils import df_to_bytes
from database import db_utils




def upload_page(conn):
    st.header("Dataset Upload")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], accept_multiple_files=False)
    if uploaded is not None:
        if uploaded.name.endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.session_state['raw_data'] = df
        st.success("Dataset uploaded and stored in session")
        st.dataframe(df.head())
        st.write(f"Shape: {df.shape}")
        st.write(preprocessing.summarize_missing(df))

        if st.button("Save uploaded dataset to database"):
            with st.spinner("Saving to database..."):
                inserted = 0
                for _, row in df.iterrows():
                    pname = row.get('product_name') or f"Product-{row.get('product_id', '')}"
                    category = row.get('category', 'Uncategorized')
                    stock = int(row.get('stock', 0)) if not pd.isna(row.get('stock', None)) else 0
                    price = float(row.get('price', 0.0)) if not pd.isna(row.get('price', None)) else 0.0
                    pid = db_utils.get_or_create_product(conn, pname, category, stock, price)
                    qty = int(row.get('quantity', 0)) if not pd.isna(row.get('quantity', None)) else 0
                    revenue = float(row.get('revenue', qty * price)) if not pd.isna(row.get('revenue', None)) else float(qty * price)
                    sale_date = str(row.get('sale_date')) if not pd.isna(row.get('sale_date', None)) else None
                    try:
                        db_utils.insert_sale(conn, pid, qty, revenue, sale_date)
                        inserted += 1
                    except Exception:
                        continue
            st.success(f"Saved {inserted} sales rows to the database")

    # quick load sample dataset button
    if st.button("Load sample dataset"):
        try:
            sample = pd.read_csv('datasets/sample_sales.csv')
            st.session_state['raw_data'] = sample
            st.success('Sample dataset loaded into session')
            st.dataframe(sample.head())
        except Exception as e:
            st.error(f"Could not load sample dataset: {e}")
        if st.button("Save sample dataset to database"):
            df = st.session_state.get('raw_data')
            if df is None:
                st.error("No sample loaded to save")
            else:
                with st.spinner("Saving sample to database..."):
                    inserted = 0
                    for _, row in df.iterrows():
                        pname = row.get('product_name') or f"Product-{row.get('product_id', '')}"
                        category = row.get('category', 'Uncategorized')
                        stock = int(row.get('stock', 0)) if not pd.isna(row.get('stock', None)) else 0
                        price = float(row.get('price', 0.0)) if not pd.isna(row.get('price', None)) else 0.0
                        pid = db_utils.get_or_create_product(conn, pname, category, stock, price)
                        qty = int(row.get('quantity', 0)) if not pd.isna(row.get('quantity', None)) else 0
                        revenue = float(row.get('revenue', qty * price)) if not pd.isna(row.get('revenue', None)) else float(qty * price)
                        sale_date = str(row.get('sale_date')) if not pd.isna(row.get('sale_date', None)) else None
                        try:
                            db_utils.insert_sale(conn, pid, qty, revenue, sale_date)
                            inserted += 1
                        except Exception:
                            continue
                st.success(f"Saved {inserted} sales rows to the database")


def preprocess_page(conn):
    st.header("Data Preprocessing")
    df = st.session_state.get('raw_data')
    if df is None:
        st.info("Upload a dataset first under Dataset Upload")
        return
    st.subheader("Original Dataset — first 100 rows")
    st.dataframe(df.head(100))

    st.subheader("Missing Values Analysis")
    miss = preprocessing.summarize_missing(df)
    st.dataframe(miss)

    st.subheader("Duplicate Records")
    dup_count = df.duplicated().sum()
    st.write(f"Duplicate rows found: {dup_count}")

    if dup_count > 0:
        st.dataframe(df[df.duplicated()])

    st.subheader("Outlier Detection (IQR) — numeric columns")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_summary = {}
    for c in num_cols:
        out = preprocessing.detect_outliers_iqr(df, c)
        outlier_summary[c] = len(out)
    st.dataframe(pd.DataFrame.from_dict(outlier_summary, orient='index', columns=['outlier_count']))

    if st.button("Run full preprocessing and clean data"):
        with st.spinner("Cleaning data..."):
            res = preprocessing.full_preprocess(df)
            cleaned = res['df']
            report = res['report']
            # treat outliers by capping for numeric cols
            for c in num_cols:
                cleaned = preprocessing.treat_outliers_iqr(cleaned, c)
            st.session_state['clean_data'] = cleaned
        st.success("Preprocessing complete")
        st.subheader("Preprocessing Report")
        st.write(report)
        st.subheader("Cleaned Dataset Preview — first 100 rows")
        st.dataframe(cleaned.head(100))
        buf = df_to_bytes(cleaned, fmt='csv')
        st.download_button("Download cleaned CSV", data=buf, file_name='cleaned_dataset.csv', mime='text/csv')
        if st.button("Save cleaned dataset to database"):
            with st.spinner("Saving cleaned dataset to DB..."):
                dfc = cleaned
                inserted = 0
                for _, row in dfc.iterrows():
                    pname = row.get('product_name') or f"Product-{row.get('product_id', '')}"
                    category = row.get('category', 'Uncategorized')
                    stock = int(row.get('stock', 0)) if not pd.isna(row.get('stock', None)) else 0
                    price = float(row.get('price', 0.0)) if not pd.isna(row.get('price', None)) else 0.0
                    pid = db_utils.get_or_create_product(conn, pname, category, stock, price)
                    qty = int(row.get('quantity', 0)) if not pd.isna(row.get('quantity', None)) else 0
                    revenue = float(row.get('revenue', qty * price)) if not pd.isna(row.get('revenue', None)) else float(qty * price)
                    sale_date = str(row.get('sale_date')) if not pd.isna(row.get('sale_date', None)) else None
                    try:
                        db_utils.insert_sale(conn, pid, qty, revenue, sale_date)
                        inserted += 1
                    except Exception:
                        continue
            st.success(f"Saved {inserted} cleaned sales rows to the database")
