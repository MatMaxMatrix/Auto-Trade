import torch
from torch import nn
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreparation:
    def __init__(self):
        self.price_scaler = MinMaxScaler(feature_range=(-1, 1))

    def prepare_data_for_model(self, df, context_length, prediction_length):
        logging.info(f"Preparing data for model: context_length={context_length}, prediction_length={prediction_length}")

        # Scale the 'close' prices
        df['scaled_close'] = self.price_scaler.fit_transform(df['close'].values.reshape(-1, 1))

        past_values = df['scaled_close'].iloc[-context_length:].values

        past_time_features = df.index[-context_length:]
        past_minute = past_time_features.minute.values
        past_hour = past_time_features.hour.values
        past_day = past_time_features.dayofweek.values

        past_minute_sin = np.sin(2 * np.pi * past_minute / 60)
        past_minute_cos = np.cos(2 * np.pi * past_minute / 60)
        past_hour_sin = np.sin(2 * np.pi * past_hour / 24)
        past_hour_cos = np.cos(2 * np.pi * past_hour / 24)
        past_day_sin = np.sin(2 * np.pi * past_day / 7)
        past_day_cos = np.cos(2 * np.pi * past_day / 7)

        time_features_combined = np.stack((past_minute_sin, past_minute_cos, past_hour_sin, past_hour_cos, past_day_sin, past_day_cos), axis=-1)
        
        return past_values, time_features_combined

    def inverse_transform(self, scaled_values):
        return self.price_scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()

class StockPredictionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = nn.Linear(7, config.embedding_dim)  # Adjusted to 7 input features
        self.xlstm = xLSTMBlockStack(config)
        self.fc = nn.Linear(config.embedding_dim, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, time_features):
        x = torch.cat([x.unsqueeze(-1), time_features], dim=-1)
        x = self.input_proj(x)
        x = self.dropout(x)
        x = self.xlstm(x)
        x = self.dropout(x)
        return self.fc(x[:, -1, :])

def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def make_prediction(model, past_values, past_time_features, device):
    with torch.no_grad():
        past_values = torch.FloatTensor(past_values).unsqueeze(0).to(device)
        past_time_features = torch.FloatTensor(past_time_features).unsqueeze(0).to(device)
        prediction = model(past_values, past_time_features)
    return prediction.cpu().numpy().flatten()

def plot_prediction(prediction, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(prediction)), prediction, label='Predicted Prices')
    plt.title('Bitcoin Price Prediction for the Next 60 Minutes')
    plt.xlabel('Minutes')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Configuration
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=8
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda",
                num_heads=8,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.5, act_fn="gelu"),
        ),
        context_length=1440,
        num_blocks=8,
        embedding_dim=256,
        slstm_at=[1, 3, 5, 7],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load the data
    bitcoin_data = pd.read_csv('bitcoin_minute_data.csv', index_col='time', parse_dates=True)
    logging.info("Loaded Bitcoin Minute-Level Data")
    logging.info(bitcoin_data.tail())

    # Prepare data for prediction
    data_prep = DataPreparation()
    past_values, past_time_features = data_prep.prepare_data_for_model(bitcoin_data, context_length=1440, prediction_length=60)

    # Load the model
    model = StockPredictionModel(cfg)
    model = load_model(model, "trained_stock_prediction_model.pth", device)
    logging.info("Model loaded successfully")

    # Make prediction
    prediction = make_prediction(model, past_values, past_time_features, device)
    
    # Inverse transform the scaled predictions
    prediction_original = data_prep.inverse_transform(prediction)
    
    # Log predictions
    logging.info(f"Predicted next 60 minutes of prices:")
    for i, price in enumerate(prediction_original):
        logging.info(f"Minute {i+1}: ${price:.2f}")

    # Plot prediction
    plot_prediction(prediction_original, 'bitcoin_price_prediction.png')
    
    logging.info("Prediction complete. Check the logs for predicted prices and 'bitcoin_price_prediction.png' for the plot.")