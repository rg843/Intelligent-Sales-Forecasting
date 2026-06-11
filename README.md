# Intelligent Sales Forecasting and Inventory Optimization System

Streamlit application for sales forecasting, inventory optimization, and BI reporting.

Features:
- Upload and persist sales datasets (CSV/XLSX)
- Automatic synthetic dataset generation (10,000 rows) if no upload
- Data preprocessing (missing values, duplicates, outlier treatment)
- Exploratory Data Analysis with interactive Plotly charts
- Supervised model training (Linear, RandomForest, GradientBoosting, XGBoost) with model persistence
- Time-series forecasting (Prophet fallback to linear/ARIMA)
- Inventory optimization (EOQ, reorder point, safety stock) with DB persistence
- Report exports: CSV, Excel, PDF (with embedded chart)

Run locally:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Notes:
- Default admin: username `admin`, password `admin` (created on first run).
- Database file: `database/database.db` (SQLite).
- To persist data to DB: use "Save to DB" buttons on Data Upload / Preprocessing pages.
- If `prophet` or `matplotlib` are not installed, the app falls back to simpler methods (linear trend, text-only PDF).

If you want, I can prepare a `docker-compose` or package the app into a single deployable container next.
