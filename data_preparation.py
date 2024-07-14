# data_preparation.py
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from os import getenv

def fetch_data_from_influx(bucket, measurement, start_time, end_time):
    load_dotenv()
    client = InfluxDBClient(url=getenv("INFLUXDB_URL"),
        username=getenv("INFLUXDB_USERNAME"),
        password=getenv("INFLUXDB_PASSWORD"),
        ssl=True, verify_ssl=True,
        org=getenv("INFLUXDB_ORG"))
    query_api = client.query_api()
    
    query = f'''
    from(bucket:"{bucket}")
    |> range(start: {start_time}, stop: {end_time})
    |> filter(fn: (r) => r._measurement == "{measurement}")
    '''
    
    result = query_api.query_data_frame(query)
    return result

def prepare_data_for_model(df, context_length, prediction_length):
    # Assuming df is sorted by time
    data = []
    for i in range(len(df) - context_length - prediction_length + 1):
        past_values = df['_value'].iloc[i:i+context_length].values
        future_values = df['_value'].iloc[i+context_length:i+context_length+prediction_length].values
        past_time_features = df['_time'].iloc[i:i+context_length].astype(int).values // 10**9  # Convert to Unix timestamp
        future_time_features = df['_time'].iloc[i+context_length:i+context_length+prediction_length].astype(int).values // 10**9
        
        data.append({
            'past_values': past_values,
            'past_time_features': past_time_features,
            'future_values': future_values,
            'future_time_features': future_time_features
        })
    
    return data

# Usage
df = fetch_data_from_influx("crypto_bucket", "BTC_price", "-30d", "now()")
prepared_data = prepare_data_for_model(df, context_length=50, prediction_length=10)
