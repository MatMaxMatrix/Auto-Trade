#%%
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

torch.cuda.empty_cache()

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}\n")

    for i in range(num_gpus):
        print(f"GPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        print()

else:
    print("CUDA is not available. No GPU found.")

# Print the total memory allocated by tensors in MB
allocated_memory = torch.cuda.memory_allocated()
print(f"Allocated memory: {allocated_memory / 1024**2:.2f} MB")

# Print the memory cached by the allocator in MB
cached_memory = torch.cuda.memory_reserved()
print(f"Cached memory: {cached_memory / 1024**2:.2f} MB")

# Get free and total memory on the GPU
free_memory = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
total_memory = torch.cuda.get_device_properties(0).total_memory

# Print out memory information
print(f"Free memory: {free_memory / 1024**2:.2f} MB")
print(f"Total memory: {total_memory / 1024**2:.2f} MB")

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
        x = checkpoint(self.xlstm, x)
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=False, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, pin_memory=False, num_workers=4)

    # Verify DataLoader
    logging.info("Verifying DataLoader...")
    for batch_idx, (past_values, past_time_features, future_values) in enumerate(train_loader):
        if batch_idx >= 2:
            break
        logging.info(f"Batch {batch_idx}: past_values shape: {past_values.shape}, past_time_features shape: {past_time_features.shape}, future_values shape: {future_values.shape}")

    # Configuration
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda",
                num_heads=4,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
        ),
        context_length=1440,
        num_blocks=4,
        embedding_dim=32,
        slstm_at=[1],

    )

    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create and train model
    model = StockPredictionModel(cfg)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count and print the number of trainable parameters
    num_params = count_parameters(model)
    print(f"Total number of trainable parameters: {num_params}")

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
