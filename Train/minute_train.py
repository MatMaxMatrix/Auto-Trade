import torch
from torch import nn
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
import pandas as pd
import numpy as np
import logging
from torch.utils.checkpoint import checkpoint
from sklearn.preprocessing import MinMaxScaler

# ... (keep the existing imports and GPU checks)

class DataPreparation:
    def __init__(self):
        self.price_scaler = MinMaxScaler(feature_range=(-1, 1))

    def prepare_data_for_model(self, df, context_length, prediction_length):
        logging.info(f"Preparing data for model: context_length={context_length}, prediction_length={prediction_length}")

        # Scale the 'close' prices
        df['scaled_close'] = self.price_scaler.fit_transform(df['close'].values.reshape(-1, 1))

        data = []
        for i in range(len(df) - context_length - prediction_length + 1):
            past_values = df['scaled_close'].iloc[i:i+context_length].values

            past_time_features = df.index[i:i+context_length]
            past_minute = past_time_features.minute.values
            past_hour = past_time_features.hour.values
            past_day = past_time_features.dayofweek.values

            past_minute_sin = np.sin(2 * np.pi * past_minute / 60)
            past_minute_cos = np.cos(2 * np.pi * past_minute / 60)
            past_hour_sin = np.sin(2 * np.pi * past_hour / 24)
            past_hour_cos = np.cos(2 * np.pi * past_hour / 24)
            past_day_sin = np.sin(2 * np.pi * past_day / 7)
            past_day_cos = np.cos(2 * np.pi * past_day / 7)

            future_values = df['scaled_close'].iloc[i+context_length:i+context_length+prediction_length].values

            time_features_combined = np.stack((past_minute_sin, past_minute_cos, past_hour_sin, past_hour_cos, past_day_sin, past_day_cos), axis=-1)
            
            data.append({
                'past_values': past_values,
                'past_time_features': time_features_combined,
                'future_values': future_values
            })

        logging.info(f"Prepared {len(data)} data points")
        return data

    def inverse_transform(self, scaled_values):
        return self.price_scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()
    
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
        x = checkpoint(self.xlstm, x)
        x = self.dropout(x)
        return self.fc(x[:, -1, :])

# ... (keep the existing StockDataset class)

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved to {save_path}")


def train_model(model, train_loader, val_loader, num_epochs, device, save_path):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    model.to(device)
    criterion.to(device)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch+1}")
        model.train()
        train_loss = 0
        for batch_idx, (past_values, past_time_features, future_values) in enumerate(train_loader):
            past_values, past_time_features, future_values = past_values.to(device), past_time_features.to(device), future_values.to(device)
            optimizer.zero_grad()
            outputs = model(past_values, past_time_features)
            loss = criterion(outputs, future_values[:, 0].unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 100 == 0:
                logging.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break

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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, pin_memory=True, num_workers=4)

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

    model = StockPredictionModel(cfg)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_params = count_parameters(model)
    print(f"Total number of trainable parameters: {num_params}")

    train_model(model, train_loader, val_loader, num_epochs=20, device=device, save_path="trained_stock_prediction_model.pth")

    # Make predictions
    model.eval()
    with torch.no_grad():
        sample = next(iter(val_loader))
        past_values, past_time_features, future_values = sample
        past_values, past_time_features, future_values = past_values.to(device), past_time_features.to(device), future_values.to(device)
        prediction = model(past_values, past_time_features)
        
        # Inverse transform the scaled predictions and actual values
        prediction_original = data_prep.inverse_transform(prediction.cpu().numpy())
        future_values_original = data_prep.inverse_transform(future_values[:, 0].cpu().numpy())
        
        logging.info(f"Predicted next price: {prediction_original[0]:.2f}")
        logging.info(f"Actual next price: {future_values_original[0]:.2f}")