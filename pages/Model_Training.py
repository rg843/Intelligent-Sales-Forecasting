import streamlit as st
import pandas as pd
from models import training
import plotly.express as px
import time
import joblib
from pathlib import Path
from database import db_utils
import json
import plotly.io as pio
import numpy as np


def app(conn):
    st.header("Model Training")

    # Diagnostic: run training using DB sales table or synthetic data
    def _run_diagnostic():
        import sqlite3
        from pathlib import Path as _Path
        import numpy as _np
        import pandas as _pd
        DB_PATH = _Path('database') / 'database.db'
        if DB_PATH.exists():
            try:
                con = sqlite3.connect(DB_PATH)
                df = _pd.read_sql_query('SELECT * FROM sales LIMIT 2000', con)
                con.close()
                # try to find a numeric target
                y_col = None
                for c in ('sales', 'amount', 'quantity', 'units'):
                    if c in df.columns:
                        y_col = c
                        break
                if y_col is not None:
                    X = df.drop(columns=[y_col]).select_dtypes(include=[_np.number]).fillna(0)
                    y = df[y_col].astype(float).fillna(0)
                    if X.shape[1] > 0:
                        return training.train_models(X, y)
            except Exception:
                pass
        # fallback synthetic
        from sklearn.datasets import make_regression
        Xv, yv = make_regression(n_samples=2000, n_features=8, noise=0.1, random_state=42)
        X = pd.DataFrame(Xv, columns=[f'f{i}' for i in range(Xv.shape[1])])
        y = pd.Series(yv)
        return training.train_models(X, y)

    df = st.session_state.get('clean_data') or st.session_state.get('raw_data')
    if df is None:
        st.info("Upload and preprocess data first")
        return

    st.write("Select features and target for supervised training")
    cols = df.columns.tolist()
    from pandas.api.types import is_numeric_dtype
    numeric_cols = [c for c in cols if is_numeric_dtype(df[c])]
    if not numeric_cols:
        st.error("No numeric columns available for model training")
        return
    target = st.selectbox("Target variable (numeric)", options=numeric_cols)
    features = st.multiselect("Features", options=[c for c in cols if c != target])

    # Run training only when the explicit button is clicked
    if st.button("Train Models"):
        if not features:
            st.error("Select at least one feature")
        else:
            X = df[features].fillna(0)
            y = df[target].fillna(0)
            st.write(f"Training with X shape={X.shape}, y shape={y.shape}")
            progress = st.progress(0)
            status = st.empty()
            for i in range(0, 50, 10):
                progress.progress(i)
                status.text(f"Preparing training... {i}%")
                time.sleep(0.05)
            try:
                with st.spinner("Training models..."):
                    results, best = training.train_models(X, y)
            except Exception as e:
                st.error('Training failed — see details below')
                st.exception(e)
                return
            progress.progress(100)
            status.text("Training complete")
            st.success(f"Best model: {best}")
            # display comparison table
            comp = pd.DataFrame(results).T.reset_index().rename(columns={'index':'model'})
            st.subheader("Model Comparison")
            st.dataframe(comp)
            # Comparison chart
            try:
                comp2 = pd.DataFrame(results).T.reset_index().rename(columns={'index':'model'})
                comp2 = comp2[['model', 'RMSE']].dropna()
                if not comp2.empty:
                    fig_comp = px.bar(comp2, x='model', y='RMSE', title='Model RMSE Comparison')
                    st.plotly_chart(fig_comp, use_container_width=True)
                    try:
                        img = fig_comp.to_image(format='png')
                        st.download_button('Download Model Comparison PNG', data=img, file_name='model_comparison.png', mime='image/png')
                    except Exception:
                        st.info('Install `kaleido` to enable PNG exports for charts.')
            except Exception:
                pass
            model_path = Path('models') / 'best_model.pkl'
            if model_path.exists():
                st.write(f"Saved best model to {model_path}")
                st.download_button('Download best model', data=model_path.read_bytes(), file_name='best_model.pkl')
                # persist model metadata
                try:
                    db_utils.insert_model(conn, best, results.get(best, {}), str(model_path))
                except Exception:
                    pass
            # store training results in session for visibility
            st.session_state['training_results'] = results
            st.session_state['best_model'] = best
            # record training run in reports table
            try:
                conn.execute('INSERT INTO reports (report_name, generated_date) VALUES (?,?)', (f"Model Training - {best}", pd.Timestamp.now().isoformat()))
                conn.commit()
            except Exception:
                pass

    # if training already ran at app startup, show results
    if 'training_results' in st.session_state and st.session_state['training_results']:
        st.subheader('Auto-run training results')
        comp = pd.DataFrame(st.session_state['training_results']).T.reset_index().rename(columns={'index':'model'})
        st.dataframe(comp)
    # show saved models from DB
    st.markdown('---')
    st.subheader('Saved Models')
    if st.button('Run Diagnostic (DB or synthetic)'):
        with st.spinner('Running diagnostic training...'):
            try:
                results_diag, best_diag = _run_diagnostic()
                st.success(f'Diagnostic best model: {best_diag}')
                compd = pd.DataFrame(results_diag).T.reset_index().rename(columns={'index':'model'})
                st.dataframe(compd)
                # prepare downloadable report (JSON + CSV) and save to exports/
                try:
                    import io
                    import datetime as _dt
                    report = {
                        'timestamp': _dt.datetime.now().isoformat(),
                        'best_model': best_diag,
                        'results': results_diag
                    }
                    json_bytes = json.dumps(report, indent=2).encode('utf-8')
                    csv_bytes = compd.to_csv(index=False).encode('utf-8')
                    exports_dir = Path('exports')
                    exports_dir.mkdir(parents=True, exist_ok=True)
                    fname_json = f'diagnostic_{_dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                    fname_csv = f'diagnostic_{_dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                    (exports_dir / fname_json).write_bytes(json_bytes)
                    (exports_dir / fname_csv).write_bytes(csv_bytes)
                    st.download_button('Download diagnostic JSON', data=json_bytes, file_name=fname_json, mime='application/json')
                    st.download_button('Download diagnostic CSV', data=csv_bytes, file_name=fname_csv, mime='text/csv')
                    st.info(f'Saved diagnostic to {exports_dir / fname_json} and {exports_dir / fname_csv}')
                except Exception:
                    st.info('Could not create downloadable diagnostic report')
            except Exception as e:
                st.error(f'Diagnostic failed: {e}')
    try:
        models_df = db_utils.get_models(conn)
        if not models_df.empty:
            st.dataframe(models_df[['model_id', 'name', 'created_at', 'metrics', 'path']].head(20))
            if st.button('Download last saved model'):
                last = models_df.iloc[0]
                p = Path(last['path'])
                if p.exists():
                    st.download_button('Download model file', data=p.read_bytes(), file_name=p.name)
                else:
                    st.error('Model file not found on disk')
        else:
            st.info('No saved models found in DB')
    except Exception:
        st.info('Could not load saved models from DB')
