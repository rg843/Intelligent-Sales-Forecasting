from flask import Flask, request, jsonify
import pandas as pd
from models import forecasting

app = Flask('ml_backend')


@app.route('/forecast', methods=['POST'])
def forecast():
    payload = request.get_json()
    # expected: data (list of records) or path to dataset
    data = payload.get('data')
    date_col = payload.get('date_col', 'ds')
    value_col = payload.get('value_col', 'y')
    periods = int(payload.get('periods', 30))
    freq = payload.get('freq', 'D')
    method = payload.get('method', 'auto')

    if data is None:
        return jsonify({'error': 'no data provided'}), 400

    df = pd.DataFrame(data)
    try:
        if method.lower() == 'prophet':
            fc = forecasting.prophet_forecast(df, date_col, value_col, periods=periods, freq=freq)
        elif method.lower() == 'arima':
            fc = forecasting.arima_forecast(df, date_col, value_col, periods=periods, freq=freq)
        elif method.lower() == 'linear':
            fc = forecasting.linear_trend_forecast(df, date_col, value_col, periods=periods, freq=freq)
        else:
            fc = forecasting.auto_forecast(df, date_col, value_col, periods=periods, freq=freq)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # serialize datetimes
    fc_out = fc.copy()
    fc_out['ds'] = fc_out['ds'].dt.strftime('%Y-%m-%d')
    return jsonify(fc_out.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8506)
