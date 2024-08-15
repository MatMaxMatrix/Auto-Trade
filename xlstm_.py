#%%
import torch
from torch import nn
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, sLSTMBlockConfig, sLSTMLayerConfig
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from os import getenv
import logging
#%%
class DataPreparation: #first function will query the data from InfluxDB, the second functoin will prepare the data in several batches based on the context length.
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
#%%
class StockPredictionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = nn.Linear(2, config.embedding_dim)  # Project 2D input to embedding_dim
        self.xlstm = xLSTMBlockStack(config)
        self.fc = nn.Linear(config.embedding_dim, 1)

    def forward(self, x, time_features):
        # Combine price and time features
        x = torch.cat([x.unsqueeze(-1), time_features.unsqueeze(-1)], dim=-1)
        # Project input to embedding dimension
        x = self.input_proj(x)
        x = self.xlstm(x)
        return self.fc(x[:, -1, :])  # Use the last time step for prediction
#%%
class StockDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.FloatTensor(item['past_values']),
            torch.FloatTensor(item['past_time_features']),
            torch.FloatTensor(item['future_values'])
        )

def train_model(model, train_loader, val_loader, num_epochs):
    criterion = nn.MSELoss()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for past_values, past_time_features, future_values in train_loader:
            optimizer.zero_grad()
            outputs = model(past_values, past_time_features)
            loss = criterion(outputs, future_values[:, 0].unsqueeze(1))  # Predict only the first future value
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for past_values, past_time_features, future_values in val_loader:
                outputs = model(past_values, past_time_features)
                val_loss += criterion(outputs, future_values[:, 0].unsqueeze(1)).item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
#%%
# Main execution
if __name__ == "__main__":
    # Fetch and prepare data
    data_prep = DataPreparation()
    prepared_data = data_prep.run_data_preparation(
        bucket="mybucket",
        measurement="binance_data",
        start_time="-30m",
        end_time="now()",
        context_length=1000,
        prediction_length=10
    )

    # Create dataset and data loaders
    dataset = StockDataset(prepared_data)
    print(len(dataset)) #is the length of the all received data - context_length - prediction_length + 1
    print(len(dataset[0])) #is the length of the first data point which is 3, past_values, past_time_features, future_values = 50, 50, 10
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    # Configuration
    cfg = xLSTMBlockStackConfig(
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="vanilla",  # Use 'vanilla' for CPU
                num_heads=50,
                conv1d_kernel_size=8,
                bias_init="powerlaw_blockdependent",
            ),
        ),
        context_length=50,  # Match your context_length
        num_blocks=500,
        embedding_dim=100,  # This should match the output of input_proj
        slstm_at=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  # Use sLSTM in all blocks
    )
    # Create and train model
    model = StockPredictionModel(cfg)
    train_model(model, train_loader, val_loader, num_epochs=50)

    # Make predictions
    model.eval()
    with torch.no_grad():
        sample = next(iter(val_loader))
        past_values, past_time_features, future_values = sample
        prediction = model(past_values, past_time_features)
        print(f"Predicted next price: {prediction[0].item()}")
        print(f"Actual next price: {future_values[0, 0].item()}")
# %%
