#%%
# data_preparation.py
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from os import getenv
import logging

class DataPreparation:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        load_dotenv()
        self.client = InfluxDBClient(
            url=getenv("INFLUXDB_URL"),
            username=getenv("INFLUXDB_USERNAME"),
            password=getenv("INFLUXDB_PASSWORD"),
            ssl=True, verify_ssl=True,
            org=getenv("INFLUXDB_ORG")
        )
        self.query_api = self.client.query_api()

    def fetch_data_from_influx(self, bucket, measurement, start_time, end_time):
        query = f'''
        from(bucket:"{bucket}")
        |> range(start: {start_time}, stop: {end_time})
        |> filter(fn: (r) => r._measurement == "{measurement}")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        result = self.query_api.query_data_frame(query)
        logging.info(f"Fetched {len(result)} rows from InfluxDB")
        logging.info(result.head())
        return result

    def prepare_data_for_model(self, df, context_length, prediction_length):
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

    def run_data_preparation(self, bucket, measurement, start_time, end_time, context_length, prediction_length):
        logging.info("Starting data preparation process")
        
        df = self.fetch_data_from_influx(bucket, measurement, start_time, end_time)
        logging.info(f"Data shape after fetching: {df.shape}")
        
        prepared_data = self.prepare_data_for_model(df, context_length, prediction_length)
        logging.info(f"Final prepared data length: {len(prepared_data)}")
        
        logging.info("Data preparation process completed")
        return prepared_data

