import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isnull().sum()
    pct = (miss / len(df)) * 100
    return pd.DataFrame({"missing": miss, "pct": pct})


def handle_missing(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ["float64", "int64"]:
                if strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "", inplace=True)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[col] < lower) | (df[col] > upper)]


def treat_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = np.where(df[col] < lower, lower, np.where(df[col] > upper, upper, df[col]))
    return df


def parse_dates(df: pd.DataFrame, col: str, fmt: str = None) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "quantity" in df.columns and "price" in df.columns:
        df["revenue"] = df["quantity"] * df["price"]
    if "sale_date" in df.columns:
        df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
        df["year"] = df["sale_date"].dt.year
        df["month"] = df["sale_date"].dt.month
        df["day"] = df["sale_date"].dt.day
    return df


def encode_labels(df: pd.DataFrame, cols: list) -> (pd.DataFrame, dict):
    mapping = {}
    df = df.copy()
    for c in cols:
        df[c], mapping[c] = pd.factorize(df[c])
    return df, mapping


def scale_features(df: pd.DataFrame, cols: list) -> (pd.DataFrame, object):
    df = df.copy()
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler


def full_preprocess(df: pd.DataFrame) -> dict:
    report = {}
    report["original_shape"] = df.shape
    report["missing_before"] = summarize_missing(df)
    df = remove_duplicates(df)
    report["after_dedup_shape"] = df.shape
    df = handle_missing(df)
    report["missing_after"] = summarize_missing(df)
    df = feature_engineer(df)
    report["final_shape"] = df.shape
    return {"df": df, "report": report}
