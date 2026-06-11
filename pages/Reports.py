import streamlit as st
import pandas as pd
from utils import df_to_bytes, format_currency
from datetime import datetime

try:
    from fpdf import FPDF
except Exception:
    FPDF = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import tempfile
import os
from database import db_utils


def app(conn):
    st.header("Reports")
    df = st.session_state.get('clean_data') or st.session_state.get('raw_data')
    if df is None:
        st.info("No data available for reports")
        return

    st.subheader("Sales Report")
    agg = df.groupby('product_name').agg({'quantity':'sum','revenue':'sum'}).reset_index() if 'product_name' in df.columns else df.head()
    agg_disp = agg.copy()
    if 'revenue' in agg_disp.columns:
        agg_disp['revenue'] = agg_disp['revenue'].apply(format_currency)
    st.dataframe(agg_disp)

    csv = df_to_bytes(agg, fmt='csv')
    st.download_button("Download Sales CSV", data=csv, file_name='sales_report.csv', mime='text/csv')

    xlsx = df_to_bytes(agg, fmt='xlsx')
    st.download_button("Download Sales Excel", data=xlsx, file_name='sales_report.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    st.subheader("Executive Summary")
    st.write("Auto-generated executive summary can be extended here.")
    if st.button('Generate PDF Report'):
        if FPDF is None:
            st.error('PDF export requires `fpdf` package. Install `pip install fpdf`')
        else:
            # attempt to create a small chart image for embedding
            chart_path = None
            try:
                if plt is not None:
                    top = agg.sort_values('revenue', ascending=False).head(10)
                    fig = plt.figure(figsize=(8, 3))
                    plt.bar(top['product_name'], top['revenue'], color='tab:blue')
                    plt.xticks(rotation=45, ha='right')
                    plt.title('Top 10 Products by Revenue')
                    plt.tight_layout()
                    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    fig.savefig(tmp.name, dpi=150)
                    plt.close(fig)
                    chart_path = tmp.name
            except Exception:
                chart_path = None

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Sales Report', ln=True)
            pdf.set_font('Arial', '', 12)
            pdf.ln(4)
            if chart_path:
                try:
                    pdf.image(chart_path, x=10, y=30, w=190)
                    pdf.ln(80)
                except Exception:
                    pass

            for i, row in agg.head(50).iterrows():
                try:
                    rev = format_currency(row['revenue']) if 'revenue' in row.index else ''
                    line = f"{row['product_name']}: qty={row['quantity']} revenue={rev}"
                except Exception:
                    line = str(row.to_dict())
                pdf.cell(0, 8, line, ln=True)

            out = pdf.output(dest='S').encode('latin-1')
            st.download_button('Download PDF', data=out, file_name=f'sales_report_{datetime.now().date()}.pdf', mime='application/pdf')

            # cleanup temp chart
            if chart_path and os.path.exists(chart_path):
                try:
                    os.unlink(chart_path)
                except Exception:
                    pass

            # record report in DB
            try:
                db_utils.insert_report(conn, 'Sales Report', datetime.now().isoformat())
            except Exception:
                pass
