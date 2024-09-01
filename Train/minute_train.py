#%%
import torch
from torch import nn
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, sLSTMBlockConfig, sLSTMLayerConfig
import pandas as pd
import numpy as np
import logging

class DataPreparation:
    def prepare_data_for_model(self, df, context_length, prediction_length):
        """
        Prepare data for model training.
        
        :param df: DataFrame with Bitcoin price data
        :param context_length: Number of past data points to use for prediction
        :param prediction_length: Number of future data points to predict
        :return: List of prepared data points
        """
        logging.info(f"Preparing data for model: context_length={context_length}, prediction_length={prediction_length}")

        data = []
        for i in range(len(df) - context_length - prediction_length + 1):
            past_values = df['close'].iloc[i:i+context_length].values
            
            # Extract and encode minute and hour as cyclical features
            past_time_features = df.index[i:i+context_length]
            past_minute = past_time_features.minute.values # [0, 1, ..., 59]
            past_hour = past_time_features.hour.values # [0, 1, ..., 23]
            
            past_minute_sin = np.sin(2 * np.pi * past_minute / 60)  #[0, 1, ..., 59] / 60 = [0, ..., 1] Or [0, 2pi]
            past_minute_cos = np.cos(2 * np.pi * past_minute / 60) #[0, 1, ..., 59] / 60 = [0, ..., 1] Or [0, 2pi]
            past_hour_sin = np.sin(2 * np.pi * past_hour / 24) # [0, 1, ..., 23] / 24 = [0, ..., 1] Or [0, 2pi]
            past_hour_cos = np.cos(2 * np.pi * past_hour / 24) # [0, 1, ..., 23] / 24 = [0, ..., 1] Or [0, 2pi]
            """
            if i < 5:  # Print only the first 5 examples
                logging.info(f"Index: {i}")
                logging.info(f"Minutes: {past_minute}")
                logging.info(f"Hours: {past_hour}")
                logging.info(f"Minute Sin: {past_minute_sin}")
                logging.info(f"Minute Cos: {past_minute_cos}")
                logging.info(f"Hour Sin: {past_hour_sin}")
                logging.info(f"Hour Cos: {past_hour_cos}")
            """
            future_values = df['close'].iloc[i+context_length:i+context_length+prediction_length].values
            
            # Combine the derived cyclical features
            time_features_combined = np.stack((past_minute_sin, past_minute_cos, past_hour_sin, past_hour_cos), axis=-1)
            if i < 2:
                logging.info(f"Time Features Combined: {time_features_combined}")
            data.append({
                'past_values': past_values,
                'past_time_features': time_features_combined,
                'future_values': future_values
            })

        logging.info(f"Prepared {len(data)} data points")
        return data

class StockPredictionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = nn.Linear(5, config.embedding_dim)  # Adjusted to 5 input features (price, sin(min), cos(min), sin(hour), cos(hour))
        self.xlstm = xLSTMBlockStack(config)
        self.fc = nn.Linear(config.embedding_dim, 1)

    def forward(self, x, time_features):
        x = torch.cat([x.unsqueeze(-1), time_features], dim=-1)  # time_features already contains four dimensions
        x = self.input_proj(x)
        x = self.xlstm(x)
        return self.fc(x[:, -1, :])

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

def train_model(model, train_loader, val_loader, num_epochs, device):
    """
    Train the model.
    
    :param model: The model to train
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for validation data
    :param num_epochs: Number of epochs to train
    :param device: Device to use for training (CPU or GPU)
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    model.to(device)
    criterion.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for past_values, past_time_features, future_values in train_loader:
            past_values, past_time_features, future_values = past_values.to(device), past_time_features.to(device), future_values.to(device)
            optimizer.zero_grad()
            outputs = model(past_values, past_time_features)
            loss = criterion(outputs, future_values[:, 0].unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for past_values, past_time_features, future_values in val_loader:
                past_values, past_time_features, future_values = past_values.to(device), past_time_features.to(device), future_values.to(device)
                outputs = model(past_values, past_time_features)
                val_loss += criterion(outputs, future_values[:, 0].unsqueeze(1)).item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        logging.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load the data
    bitcoin_data = pd.read_csv('bitcoin_minute_data.csv', index_col='time', parse_dates=True)
    logging.info("Loaded Bitcoin Minute-Level Data")
    logging.info(bitcoin_data.tail())
    assert 'close' in bitcoin_data.columns, "The 'close' column is missing in the loaded data."

    # Prepare data for model
    data_prep = DataPreparation()
    prepared_data = data_prep.prepare_data_for_model(bitcoin_data, context_length=1440, prediction_length=60)
    logging.info("Data is prepared")

    # Create dataset and data loaders
    dataset = StockDataset(prepared_data)
    logging.info(f"Dataset is created with {len(dataset)} data points")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    # Configuration
    cfg = xLSTMBlockStackConfig(
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="vanilla",
                num_heads=8,
                conv1d_kernel_size=3,
                bias_init="powerlaw_blockdependent",
            ),
        ),
        context_length=1000,
        num_blocks=4,
        embedding_dim=64,
        slstm_at=[0, 1, 2, 3],
    )

    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create and train model
    model = StockPredictionModel(cfg)
    train_model(model, train_loader, val_loader, num_epochs=50, device=device)

    # Make predictions
    model.eval()
    with torch.no_grad():
        sample = next(iter(val_loader))
        past_values, past_time_features, future_values = sample
        past_values, past_time_features, future_values = past_values.to(device), past_time_features.to(device), future_values.to(device)
        prediction = model(past_values, past_time_features)
        logging.info(f"Predicted next price: {prediction[0].item()}")
        logging.info(f"Actual next price: {future_values[0, 0].item()}")
