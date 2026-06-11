import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from models import forecasting
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from database import db_utils
import plotly.io as pio



def app(conn):
    st.header("Sales Forecasting")
    df = st.session_state.get('clean_data') or st.session_state.get('raw_data')
    if df is None:
        st.info("Upload data first")
        return

    if 'sale_date' not in df.columns or 'quantity' not in df.columns:
        st.error("Dataset must contain 'sale_date' and 'quantity' columns for forecasting")
        return
    # Hyperparameters
    col1, col2 = st.columns(2)
    with col1:
        horizon = st.selectbox("Forecast Horizon", ['7 Days', '30 Days', '90 Days', '6 Months', '1 Year'])
        mapping = {'7 Days':7, '30 Days':30, '90 Days':90, '6 Months':180, '1 Year':365}
        periods = mapping[horizon]
        freq = st.selectbox('Frequency', ['D', 'W', 'M'])
    with col2:
        method = st.selectbox('Method', ['Auto', 'Prophet', 'ARIMA', 'Linear'])
        arima_seasonal = st.checkbox('ARIMA seasonal', value=False)
        prophet_weekly = st.checkbox('Prophet weekly seasonality', value=True)

    if st.button("Run Forecast"):
        st.info(f'Running forecasting pipeline using {method}')
        try:
            if method == 'Prophet':
                fc = forecasting.prophet_forecast(df, 'sale_date', 'quantity', periods=periods, freq=freq)
            elif method == 'ARIMA':
                fc = forecasting.arima_forecast(df, 'sale_date', 'quantity', periods=periods, freq=freq)
            elif method == 'Linear':
                fc = forecasting.linear_trend_forecast(df, 'sale_date', 'quantity', periods=periods, freq=freq)
            else:
                fc = forecasting.auto_forecast(df, 'sale_date', 'quantity', periods=periods, freq=freq)
        except Exception as e:
            st.error(f'Forecasting failed: {e}')
            return
        st.subheader("Forecast Results")
        st.dataframe(fc.tail())

        fig = go.Figure()
        hist = df.groupby(pd.to_datetime(df['sale_date']).dt.date).agg({'quantity':'sum'}).reset_index()
        fig.add_trace(go.Scatter(x=hist['sale_date'], y=hist['quantity'], name='Historical'))
        fig.add_trace(go.Scatter(x=fc['ds'], y=fc['predicted_sales'], name='Forecast'))
        if 'lower' in fc.columns and 'upper' in fc.columns:
            fig.add_trace(go.Scatter(x=fc['ds'], y=fc['lower'], name='Lower', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=fc['ds'], y=fc['upper'], name='Upper', line=dict(dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

        try:
            img = fig.to_image(format='png')
            st.download_button('Download Forecast PNG', data=img, file_name='sales_forecast.png', mime='image/png')
        except Exception:
            st.info('Install `kaleido` to enable PNG exports for charts.')
        csv = fc.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast CSV", csv, file_name='forecast.csv', mime='text/csv')

        # persist forecasts to DB (product-level if product selected)
        try:
            prod_sel = st.selectbox('Product to attach forecast', options=['All'] + (df['product_name'].unique().tolist() if 'product_name' in df.columns else []))
            pid = None
            if prod_sel != 'All' and 'product_name' in df.columns:
                # get or create product id
                # try to find product id from products table first
                try:
                    prods = db_utils.get_products(conn)
                    row = prods[prods['product_name'] == prod_sel]
                    if not row.empty:
                        pid = int(row.iloc[0]['product_id'])
                    else:
                        pid = db_utils.get_or_create_product(conn, prod_sel, 'Uncategorized', 0, 0.0)
                except Exception:
                    pid = db_utils.get_or_create_product(conn, prod_sel, 'Uncategorized', 0, 0.0)

            saved = 0
            for _, r in fc.iterrows():
                try:
                    ds = r.get('ds') if 'ds' in r else r.index
                    val = float(r.get('predicted_sales') if 'predicted_sales' in r else r.iloc[1])
                    db_utils.insert_forecast(conn, pid, str(ds), val)
                    saved += 1
                except Exception:
                    continue
            st.success(f"Saved {saved} forecast rows to the database")
        except Exception:
            st.info('Could not persist forecasts to DB')

    # auto-run forecast when daily aggregation exists
    if 'daily_agg' in st.session_state and st.session_state['daily_agg'] is not None:
        if st.button('Auto Forecast (daily agg)'):
            daily = st.session_state['daily_agg']
            try:
                fc = forecasting.prophet_forecast(daily.rename(columns={'ds':'ds','y':'y'}), 'ds', 'y', periods=periods)
            except Exception:
                # fallback linear
                import numpy as np
                from sklearn.linear_model import LinearRegression
                from datetime import timedelta
                tmp = daily.copy()
                tmp['ds'] = pd.to_datetime(tmp['ds'])
                tmp['t'] = (tmp['ds'] - tmp['ds'].min()).dt.days
                model = LinearRegression()
                model.fit(tmp[['t']], tmp['y'])
                last = tmp['ds'].max()
                future = [last + timedelta(days=i) for i in range(1, periods+1)]
                t_future = np.array([(d - tmp['ds'].min()).days for d in future]).reshape(-1,1)
                preds = model.predict(t_future)
                fc = pd.DataFrame({'ds': future, 'predicted_sales': preds, 'lower': preds - preds.std(), 'upper': preds + preds.std()})
            st.subheader('Auto Forecast Results')
            st.dataframe(fc.tail())
            fig = go.Figure()
            hist = df.groupby(pd.to_datetime(df['sale_date']).dt.date).agg({'quantity':'sum'}).reset_index()
            fig.add_trace(go.Scatter(x=hist['sale_date'], y=hist['quantity'], name='Historical'))
            fig.add_trace(go.Scatter(x=fc['ds'], y=fc.iloc[:,1], name='Forecast'))
            st.plotly_chart(fig, use_container_width=True)
