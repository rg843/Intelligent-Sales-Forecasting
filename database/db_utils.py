import sqlite3
from typing import Optional
import pandas as pd
import utils


def create_user(conn: sqlite3.Connection, username: str, email: str, password: str) -> bool:
    """Create a new user. `password` is plaintext and will be hashed.
    Returns True on success, False on failure.
    """
    try:
        cur = conn.cursor()
        ph = utils.hash_password(password)
        cur.execute("INSERT INTO users (username,email,password) VALUES (?,?,?)", (username, email, ph))
        conn.commit()
        return True
    except Exception:
        return False


def authenticate(conn: sqlite3.Connection, username: str, password: str) -> Optional[dict]:
    """Authenticate a user by plaintext password. Returns user dict or None."""
    cur = conn.cursor()
    cur.execute("SELECT user_id,username,email,password FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    if row:
        stored = row[3]
        if utils.verify_password(password, stored):
            return {"user_id": row[0], "username": row[1], "email": row[2]}
    return None


def insert_product(conn: sqlite3.Connection, name: str, category: str, stock: int, price: float):
    cur = conn.cursor()
    cur.execute("INSERT INTO products (product_name,category,stock,price) VALUES (?,?,?,?)", (name, category, stock, price))
    conn.commit()
    return cur.lastrowid


def get_products(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM products", conn)


def get_or_create_product(conn: sqlite3.Connection, name: str, category: str, stock: int, price: float) -> int:
    """Return product_id for existing product by name or create it and return new id."""
    cur = conn.cursor()
    cur.execute("SELECT product_id FROM products WHERE product_name=?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    return insert_product(conn, name, category, stock, price)


def insert_sale(conn: sqlite3.Connection, product_id: int, quantity: int, revenue: float, sale_date: str):
    cur = conn.cursor()
    cur.execute("INSERT INTO sales (product_id,quantity,revenue,sale_date) VALUES (?,?,?,?)", (product_id, quantity, revenue, sale_date))
    conn.commit()
    return cur.lastrowid


def get_sales(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM sales", conn)


def insert_forecast(conn: sqlite3.Connection, product_id: int | None, forecast_date: str, predicted_sales: float):
    cur = conn.cursor()
    cur.execute("INSERT INTO forecasts (product_id,forecast_date,predicted_sales) VALUES (?,?,?)", (product_id, forecast_date, predicted_sales))
    conn.commit()
    return cur.lastrowid


def get_forecasts(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM forecasts", conn)


def insert_report(conn: sqlite3.Connection, report_name: str, generated_date: str):
    cur = conn.cursor()
    cur.execute("INSERT INTO reports (report_name,generated_date) VALUES (?,?)", (report_name, generated_date))
    conn.commit()
    return cur.lastrowid


def get_reports(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM reports", conn)


def insert_inventory(conn: sqlite3.Connection, product_id: int, stock_level: int, reorder_point: int, safety_stock: int):
    cur = conn.cursor()
    cur.execute("INSERT INTO inventory (product_id, stock_level, reorder_point, safety_stock) VALUES (?,?,?,?)",
                (product_id, stock_level, reorder_point, safety_stock))
    conn.commit()
    return cur.lastrowid


def upsert_inventory(conn: sqlite3.Connection, product_id: int, stock_level: int, reorder_point: int, safety_stock: int):
    cur = conn.cursor()
    # try update first
    cur.execute("SELECT inventory_id FROM inventory WHERE product_id=?", (product_id,))
    row = cur.fetchone()
    if row:
        cur.execute("UPDATE inventory SET stock_level=?, reorder_point=?, safety_stock=? WHERE product_id=?",
                    (stock_level, reorder_point, safety_stock, product_id))
        conn.commit()
        return row[0]
    else:
        return insert_inventory(conn, product_id, stock_level, reorder_point, safety_stock)


def get_inventory(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM inventory", conn)


def insert_model(conn: sqlite3.Connection, name: str, metrics: dict, path: str) -> int:
    import json
    cur = conn.cursor()
    metrics_json = json.dumps(metrics)
    cur.execute("INSERT INTO models (name, created_at, metrics, path) VALUES (?,?,?,?)", (name, pd.Timestamp.now().isoformat(), metrics_json, path))
    conn.commit()
    return cur.lastrowid


def get_models(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM models ORDER BY created_at DESC", conn)


def insert_saved_view(conn: sqlite3.Connection, name: str, username: str, config: str) -> int:
    cur = conn.cursor()
    cur.execute("INSERT INTO saved_views (name, username, config, created_at) VALUES (?,?,?,?)",
                (name, username, config, pd.Timestamp.now().isoformat()))
    conn.commit()
    return cur.lastrowid


def get_saved_views(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        return pd.read_sql_query("SELECT * FROM saved_views ORDER BY created_at DESC", conn)
    except Exception:
        return pd.DataFrame()


def delete_saved_view(conn: sqlite3.Connection, view_id: int) -> bool:
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM saved_views WHERE view_id=?", (view_id,))
        conn.commit()
        return True
    except Exception:
        return False
