import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    # Use sqrt of MSE for RMSE to be compatible with older sklearn versions
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape), "R2": float(r2)}


def _prepare_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """One-hot encode categorical columns and align train/test frames."""
    # Convert object/category dtypes to dummies
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'object' or str(X_train[c].dtype).startswith('category')]
    if not cat_cols:
        return X_train.copy(), X_test.copy()

    X_train_d = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
    X_test_d = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)

    # align columns
    X_train_d, X_test_d = X_train_d.align(X_test_d, join='outer', axis=1, fill_value=0)
    return X_train_d.fillna(0), X_test_d.fillna(0)


def train_models(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Train several regressors, evaluate on a hold-out set, and save the best model.

    Returns: (results_dict, best_model_name)
    """
    if X is None or y is None:
        raise ValueError("X and y must be provided")

    # Ensure inputs are DataFrames/Series
    X = pd.DataFrame(X)
    y = pd.Series(y).astype(float)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # prepare features (handle categorical variables)
    X_train_p, X_test_p = _prepare_features(X_train, X_test)

    results = {}
    models = {}

    # define candidate models
    candidates = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
    }

    if XGBRegressor is not None:
        candidates['XGBoost'] = XGBRegressor(n_estimators=100, random_state=random_state, verbosity=0)

    # train and evaluate on hold-out test
    for name, model in candidates.items():
        try:
            model.fit(X_train_p, y_train)
            pred = model.predict(X_test_p)
            results[name] = regression_metrics(y_test.values, pred)
            models[name] = model
        except Exception as e:
            results[name] = {"error": str(e)}

    # choose best by RMSE (ignore models that errored)
    valid = {n: m for n, m in results.items() if isinstance(m, dict) and 'RMSE' in m}
    if not valid:
        raise RuntimeError("All candidate models failed during training/evaluation")

    best = min(valid.items(), key=lambda x: x[1]['RMSE'])[0]

    # Retrain best on full dataset (prepare full features)
    X_full = pd.get_dummies(X, columns=[c for c in X.columns if X[c].dtype == 'object' or str(X[c].dtype).startswith('category')], drop_first=False).fillna(0)
    # Align columns with trained model if necessary
    model_best = models[best]
    try:
        model_best.fit(X_full, y)
    except Exception:
        # fallback: fit on training portion
        model_best.fit(pd.concat([X_train_p, X_test_p], axis=0), pd.concat([y_train, y_test], axis=0))

    # persist best model
    Path('models').mkdir(parents=True, exist_ok=True)
    model_path = Path('models') / 'best_model.pkl'
    joblib.dump(model_best, model_path)

    return results, best
