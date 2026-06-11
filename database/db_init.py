import sqlite3
from pathlib import Path

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS users (user_id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, email TEXT UNIQUE, password TEXT)",
    "CREATE TABLE IF NOT EXISTS products (product_id INTEGER PRIMARY KEY AUTOINCREMENT, product_name TEXT, category TEXT, stock INTEGER, price REAL)",
    "CREATE TABLE IF NOT EXISTS sales (sale_id INTEGER PRIMARY KEY AUTOINCREMENT, product_id INTEGER, quantity INTEGER, revenue REAL, sale_date TEXT)",
    "CREATE TABLE IF NOT EXISTS inventory (inventory_id INTEGER PRIMARY KEY AUTOINCREMENT, product_id INTEGER, stock_level INTEGER, reorder_point INTEGER, safety_stock INTEGER)",
    "CREATE TABLE IF NOT EXISTS forecasts (forecast_id INTEGER PRIMARY KEY AUTOINCREMENT, product_id INTEGER, forecast_date TEXT, predicted_sales REAL)",
    "CREATE TABLE IF NOT EXISTS reports (report_id INTEGER PRIMARY KEY AUTOINCREMENT, report_name TEXT, generated_date TEXT)",
    "CREATE TABLE IF NOT EXISTS models (model_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, created_at TEXT, metrics TEXT, path TEXT)",
    "CREATE TABLE IF NOT EXISTS saved_views (view_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, username TEXT, config TEXT, created_at TEXT)",
]


def init_db(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    for stmt in SCHEMA:
        cur.execute(stmt)
    conn.commit()
    # create default admin user for testing if not exists
    try:
        from . import db_utils
        cur.execute("SELECT 1 FROM users WHERE username='admin'")
        if cur.fetchone() is None:
            # create with plaintext handled by db_utils
            db_utils.create_user(conn, 'admin', 'admin@example.com', 'admin')
            conn.commit()
    except Exception:
        pass
    conn.close()
