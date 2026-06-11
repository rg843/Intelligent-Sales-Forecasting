import math
import pandas as pd


def economic_order_quantity(demand, order_cost, holding_cost):
    try:
        return math.sqrt((2 * demand * order_cost) / holding_cost)
    except Exception:
        return None


def safety_stock(service_level_sigma, lead_time_sd):
    return service_level_sigma * lead_time_sd


def reorder_point(avg_daily_demand, lead_time_days, safety_stock_val):
    return int(avg_daily_demand * lead_time_days + safety_stock_val)


def inventory_turnover(cogs, avg_inventory):
    try:
        return cogs / avg_inventory
    except Exception:
        return None


def stock_coverage_days(avg_daily_sales, stock_level):
    if avg_daily_sales == 0:
        return float('inf')
    return stock_level / avg_daily_sales


def compute_inventory_metrics(df_products: pd.DataFrame, df_sales: pd.DataFrame) -> pd.DataFrame:
    df = df_products.copy()
    sales_sum = df_sales.groupby('product_id').agg({'quantity': 'sum', 'revenue': 'sum'}).reset_index()
    df = df.merge(sales_sum, left_on='product_id', right_on='product_id', how='left')
    df['quantity'] = df['quantity'].fillna(0)
    df['revenue'] = df['revenue'].fillna(0)
    df['avg_daily_sales'] = df['quantity'] / 30
    df['eoq'] = df.apply(lambda r: economic_order_quantity(r['quantity'] if r['quantity']>0 else 1, order_cost=50, holding_cost= r['price']*0.2 if r['price']>0 else 1), axis=1)
    df['reorder_point'] = df.apply(lambda r: reorder_point(r['avg_daily_sales'], lead_time_days=7, safety_stock_val=10), axis=1)
    df['stock_coverage_days'] = df.apply(lambda r: stock_coverage_days(r['avg_daily_sales'], r['stock']), axis=1)
    return df
