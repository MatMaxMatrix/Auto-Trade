#%%
# data_preparation.py
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from os import getenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")

    '''
    
    result = query_api.query_data_frame(query)
    logging.info(f"Fetched {len(result)} rows from InfluxDB")
    logging.info(result.head())
    return result
#%%
def prepare_data_for_model(df, context_length, prediction_length):
    logging.info(f"Preparing data for model: context_length={context_length}, prediction_length={prediction_length}")

    data = []
    for i in range(len(df) - context_length - prediction_length + 1):
        past_values = df['close_price'].iloc[i:i+context_length].values
        future_values = df['close_price'].iloc[i+context_length:i+context_length+prediction_length].values
        past_time_features = df['_time'].iloc[i:i+context_length].astype(int).values // 10**9  # Convert to Unix timestamp
        future_time_features = df['_time'].iloc[i+context_length:i+context_length+prediction_length].astype(int).values // 10**9
        
        data.append({
            'past_values': past_values,
            'past_time_features': past_time_features,
            'future_values': future_values,
            'future_time_features': future_time_features
        })
    logging.info(f"Prepared {len(data)} data points")
    return data
#%%
# Usage
logging.info("Starting data preparation process")
#%%
df = fetch_data_from_influx("mybucket", "binance_data", "-2m", "now()")
#%%
logging.info(f"Data shape after fetching: {df.shape}")
#%%
prepared_data = prepare_data_for_model(df, context_length=50, prediction_length=10)
#%%
logging.info(f"Final prepared data length: {len(prepared_data)}")
#%%
logging.info("Data preparation process completed")
# %%
print(prepared_data[0])
# %%
