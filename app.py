import streamlit as st
from pathlib import Path
import sqlite3
import pandas as pd
import datetime as dt

from database import db_init, db_utils
import utils


DB_PATH = Path(__file__).parent / "database" / "database.db"
db_init.init_db(str(DB_PATH))
utils.inject_custom_css()


def _login(conn):
    st.header("Login")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login"):
        user = db_utils.authenticate(conn, username, password)
        if user:
            st.session_state.user = user
            st.session_state['last_active'] = dt.datetime.now().isoformat()
            st.success(f"Welcome {username}")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")


def _register(conn):
    st.header("Register")
    username = st.text_input("Username", key="reg_user")
    email = st.text_input("Email", key="reg_email")
    password = st.text_input("Password", type="password", key="reg_pass")
    if st.button("Create Account"):
        if not username or not email or not password:
            st.error("Please fill all fields")
            return
        ok = db_utils.create_user(conn, username, email, password)
        if ok:
            st.success("Account created — please login")
        else:
            st.error("Registration failed — username or email may already exist")


def main():
    st.set_page_config(page_title="Intelligent Sales Forecasting", layout="wide")

    if "user" not in st.session_state:
        st.session_state.user = None

    # auto-generate synthetic dataset when none uploaded
    if "raw_data" not in st.session_state:
        try:
            st.session_state['raw_data'] = utils.generate_synthetic_sales(10000)
            st.session_state['generated'] = True
        except Exception:
            st.session_state['raw_data'] = None

    # session timeout handling (auto-logout after inactivity)
    try:
        timeout_seconds = 60 * 60  # 1 hour
        last = st.session_state.get('last_active')
        if last and st.session_state.get('user'):
            try:
                last_dt = dt.datetime.fromisoformat(last)
                if (dt.datetime.now() - last_dt).total_seconds() > timeout_seconds:
                    st.session_state.user = None
                    st.warning('Session expired due to inactivity. Please log in again.')
            except Exception:
                pass
    except Exception:
        pass

    st.sidebar.title("Intelligent Sales Forecasting")
    demo_mode = st.sidebar.checkbox('Demo mode (auto-login)', value=True)
    base_pages = [
        "Home",
        "Data Upload",
        "Data Preprocessing",
        "EDA",
        "Model Training",
        "Sales Forecasting",
        "Inventory Optimization",
        "Reports",
        "Dashboard",
    ]

    if st.session_state.user:
        pages = base_pages + ["Logout"]
    else:
        pages = ["Login", "Register"] + base_pages

    choice = st.sidebar.selectbox("Navigation", pages)

    # Demo auto-login to speed development/testing and avoid blank pages
    if demo_mode and not st.session_state.get('user'):
        try:
            # attempt to authenticate admin user from DB
            cur = conn.cursor()
            cur.execute("SELECT user_id,username,email FROM users WHERE username='admin'")
            row = cur.fetchone()
            if row:
                st.session_state.user = {"user_id": row[0], "username": row[1], "email": row[2]}
        except Exception:
            # fallback to a simple demo user
            st.session_state.user = {"user_id": 0, "username": "demo", "email": "demo@example.com"}

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    try:
        if choice == "Login":
            _login(conn)
        elif choice == "Register":
            _register(conn)
        elif choice == "Logout":
            st.session_state.user = None
            st.experimental_rerun()
        else:
            # protected pages
            if not st.session_state.user:
                st.info("Please login or register to continue.")
                return

            # Route to pages
            if choice == "Home":
                from pages.Home import app as home_app
                home_app(conn)
            elif choice == "Data Upload":
                from pages.Data_Preprocessing import upload_page
                upload_page(conn)
            elif choice == "Data Preprocessing":
                from pages.Data_Preprocessing import preprocess_page
                preprocess_page(conn)
            elif choice == "EDA":
                from pages.EDA import app as eda_app
                eda_app(conn)
            elif choice == "Model Training":
                from pages.Model_Training import app as mt_app
                mt_app(conn)
            elif choice == "Sales Forecasting":
                from pages.Sales_Forecasting import app as sf_app
                sf_app(conn)
            elif choice == "Inventory Optimization":
                from pages.Inventory_Optimization import app as io_app
                io_app(conn)
            elif choice == "Reports":
                from pages.Reports import app as reports_app
                reports_app(conn)
            elif choice == "Dashboard":
                from pages.Dashboard import app as dash_app
                dash_app(conn)
    finally:
        # update last_active timestamp for session activity
        try:
            if st.session_state.get('user'):
                st.session_state['last_active'] = dt.datetime.now().isoformat()
        except Exception:
            pass
        conn.close()


if __name__ == "__main__":
    main()
