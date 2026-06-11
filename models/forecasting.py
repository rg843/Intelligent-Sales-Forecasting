import pandas as pd
import numpy as np
from pathlib import Path

try:
    from prophet import Prophet
except Exception:
    Prophet = None
try:
    from pmdarima import auto_arima
except Exception:
    auto_arima = None


def prophet_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 30, freq: str = "D") -> pd.DataFrame:
    if Prophet is None:
        raise ImportError("Prophet library not available. Install `prophet` package.")

    df2 = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"}).dropna()
    m = Prophet()
    m.fit(df2)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    fcst = m.predict(future)
    out = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"yhat": "predicted_sales", "yhat_lower": "lower", "yhat_upper": "upper"})
    return out


def arima_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
    if auto_arima is None:
        raise ImportError('pmdarima not available')
    tmp = df[[date_col, value_col]].dropna().copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp = tmp.sort_values(date_col)
    # aggregate to frequency (daily) by summing
    ts = tmp.set_index(date_col)[value_col].resample(freq).sum().fillna(0)
    # fit auto_arima
    model = auto_arima(ts, seasonal=False, error_action='ignore', suppress_warnings=True)
    fc = model.predict(n_periods=periods)
    last = ts.index.max()
    future_index = pd.date_range(start=last + pd.Timedelta(1, unit=freq), periods=periods, freq=freq)
    out = pd.DataFrame({'ds': future_index, 'predicted_sales': fc})
    # approximate intervals using simple std dev
    out['lower'] = out['predicted_sales'] - np.std(fc)
    out['upper'] = out['predicted_sales'] + np.std(fc)
    return out


def linear_trend_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression
    tmp = df[[date_col, value_col]].dropna().copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp = tmp.sort_values(date_col)
    tmp['t'] = (tmp[date_col] - tmp[date_col].min()).dt.days
    X = tmp[['t']].values.reshape(-1, 1)
    y = tmp[value_col].values
    if len(X) < 2:
        # not enough data, return zeros
        last = tmp[date_col].max()
        future = pd.date_range(start=last + pd.Timedelta(1, unit=freq), periods=periods, freq=freq)
        return pd.DataFrame({'ds': future, 'predicted_sales': np.zeros(periods), 'lower': np.zeros(periods), 'upper': np.zeros(periods)})
    model = LinearRegression()
    model.fit(X, y)
    last = tmp[date_col].max()
    future_dates = pd.date_range(start=last + pd.Timedelta(1, unit=freq), periods=periods, freq=freq)
    t_future = np.array([(d - tmp[date_col].min()).days for d in future_dates]).reshape(-1, 1)
    preds = model.predict(t_future)
    return pd.DataFrame({'ds': future_dates, 'predicted_sales': preds, 'lower': preds - np.std(preds), 'upper': preds + np.std(preds)})


def auto_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
    """Attempt Prophet, then ARIMA, then linear trend as fallback. Returns standardized forecast DataFrame."""
    # try Prophet
    try:
        return prophet_forecast(df, date_col, value_col, periods=periods, freq=freq)
    except Exception:
        pass
    # try ARIMA
    try:
        return arima_forecast(df, date_col, value_col, periods=periods, freq=freq)
    except Exception:
        pass
    # fallback to linear
    return linear_trend_forecast(df, date_col, value_col, periods=periods, freq=freq)
