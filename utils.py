import streamlit as st
import hashlib
import sqlite3
from io import BytesIO
import pandas as pd


def inject_custom_css():
    css = """
    <style>
    /* App background and typography */
    .stApp { background-color: #f7f9fc; }
    .big-font { font-size:20px !important; }
    /* Tidy header and footer for a cleaner BI look */
    header[role="banner"] { display: none !important; }
    footer { visibility: hidden; }
    /* Wider containers */
    .css-1d391kg { max-width: 1200px; margin: 0 auto; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def hash_password(password: str, iterations: int = 100_000) -> str:
    """Create a salted PBKDF2-HMAC-SHA256 password hash.

    Stored format: pbkdf2_sha256$iterations$salt_hex$hash_hex
    """
    import os
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Verify a plaintext password against the stored hash."""
    try:
        if stored.startswith('pbkdf2_sha256$'):
            _, it, salt_hex, hash_hex = stored.split('$')
            iterations = int(it)
            salt = bytes.fromhex(salt_hex)
            expected = bytes.fromhex(hash_hex)
            dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations)
            return hashlib.compare_digest(dk, expected)
        # fallback to legacy sha256
        return _hash_password(password) == stored
    except Exception:
        return False


def create_user(conn: sqlite3.Connection, username: str, email: str, password: str) -> bool:
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username,email,password) VALUES (?,?,?)", (username, email, _hash_password(password)))
        conn.commit()
        return True
    except Exception:
        return False


def authenticate(conn: sqlite3.Connection, username: str, password: str):
    cur = conn.cursor()
    cur.execute("SELECT user_id,username,email FROM users WHERE username=? AND password=?", (username, _hash_password(password)))
    row = cur.fetchone()
    if row:
        return {"user_id": row[0], "username": row[1], "email": row[2]}
    return None


def df_to_bytes(df: pd.DataFrame, fmt: str = "csv") -> bytes:
    buf = BytesIO()
    if fmt == "csv":
        df.to_csv(buf, index=False)
    else:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
    return buf.getvalue()


def format_currency(value, currency_symbol=None, decimals=2):
    """Format a numeric value with a currency symbol.

    If `currency_symbol` is None, prefer a Streamlit session value
    at `st.session_state['currency_symbol']` if available, otherwise
    fall back to the generic currency sign '¤'. The Streamlit import
    is done lazily so this helper can be used outside a Streamlit context.
    """
    if value is None:
        return ""
    try:
        val = float(value)
    except Exception:
        return value

    if currency_symbol is None:
        try:
            import streamlit as st
            currency_symbol = st.session_state.get('currency_symbol', '¤')
        except Exception:
            currency_symbol = '¤'

    try:
        return f"{currency_symbol}{val:,.{decimals}f}"
    except Exception:
        return f"{currency_symbol}{val}"


def generate_synthetic_sales(n: int = 10000, start_date: str = "2024-01-01") -> pd.DataFrame:
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    np.random.seed(42)
    products = [
        (1, 'Widget A', 'Widgets', 100, 19.99),
        (2, 'Gadget B', 'Gadgets', 80, 49.50),
        (3, 'Thingamajig C', 'Things', 50, 149.99),
        (4, 'Widget D', 'Widgets', 200, 29.99),
        (5, 'Accessory E', 'Accessories', 500, 5.99),
    ]

    start = datetime.fromisoformat(start_date)
    rows = []
    for i in range(1, n + 1):
        pid, pname, category, stock, price = products[np.random.randint(0, len(products))]
        qty = int(np.random.poisson(3)) + 1
        revenue = round(qty * price, 2)
        delta = np.random.randint(0, 365)
        sale_date = (start + timedelta(days=int(delta))).date().isoformat()
        region = np.random.choice(['North', 'South', 'East', 'West'])
        customer_id = 1000 + np.random.randint(0, 1000)
        rows.append({
            'sale_id': i,
            'product_id': pid,
            'product_name': pname,
            'category': category,
            'quantity': qty,
            'price': price,
            'revenue': revenue,
            'sale_date': sale_date,
            'stock': stock,
            'region': region,
            'customer_id': customer_id,
        })

    df = pd.DataFrame(rows)
    return df
