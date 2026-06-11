"""Simple runner to test the model training pipeline.
Usage: run directly in project root. It will attempt to load sales training data
from the database; if not found, it generates a synthetic regression dataset.
"""
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from models import training


DB_PATH = Path('database') / 'database.db'


def load_from_db():
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    try:
        # try to load a table that looks like sales with numeric target 'quantity' or 'sales'
        for col in ('quantity', 'sales', 'amount', 'units'):
            try:
                df = pd.read_sql_query(f"SELECT * FROM sales LIMIT 1000", conn)
                if col in df.columns:
                    return df
            except Exception:
                continue
    finally:
        conn.close()
    return None


def prepare_xy_from_df(df):
    # Heuristic: if there's a 'sales' or 'quantity' column, use it as y and others as X
    y_col = None
    for c in ('sales', 'amount', 'quantity', 'units'):
        if c in df.columns:
            y_col = c
            break
    if y_col is None:
        return None, None
    y = df[y_col].astype(float)
    X = df.drop(columns=[y_col])
    # drop non-numeric-ish columns
    X = X.select_dtypes(include=[np.number])
    if X.shape[0] == 0 or X.shape[1] == 0:
        return None, None
    return X, y


def main():
    df = load_from_db()
    if df is None:
        print('No DB sales table available or suitable target column; generating synthetic dataset...')
        from sklearn.datasets import make_regression
        Xv, yv = make_regression(n_samples=2000, n_features=8, noise=0.1, random_state=42)
        X = pd.DataFrame(Xv, columns=[f'f{i}' for i in range(Xv.shape[1])])
        y = pd.Series(yv)
    else:
        print('Loaded data from DB; preparing X/y')
        X, y = prepare_xy_from_df(df)
        if X is None or y is None:
            print('DB data could not be used for training; falling back to synthetic')
            from sklearn.datasets import make_regression
            Xv, yv = make_regression(n_samples=2000, n_features=8, noise=0.1, random_state=42)
            X = pd.DataFrame(Xv, columns=[f'f{i}' for i in range(Xv.shape[1])])
            y = pd.Series(yv)

    print('Running training...')
    results, best = training.train_models(X, y)
    print('Training results:')
    for k, v in results.items():
        print(k, v)
    print('Best model:', best)
    model_path = Path('models') / 'best_model.pkl'
    print('Saved model path:', model_path, 'exists=', model_path.exists())


if __name__ == '__main__':
    main()
