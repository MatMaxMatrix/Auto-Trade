import torch
from torch import nn
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from datetime import datetime
import sys
from typing import Dict, Any, Tuple, Optional
import json
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from tqdm import tqdm

# Data preparation classes and functions
@dataclass
class PreparedDataInfo:
    """Stores metadata about the prepared data"""
    context_length: int
    prediction_length: int
    train_size: int
    val_size: int
    feature_dim: int
    data_hash: str
    scaler_params: Dict[str, Any]
    preparation_date: str
    original_data_shape: Tuple[int, int]

class DataPreparation:
    def __init__(self, cache_dir='data_cache'):
        self.price_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def prepare_and_save(self, df: pd.DataFrame, context_length: int, prediction_length: int, 
                        save_path: str, train_split: float = 0.8) -> PreparedDataInfo:
        """Prepare the data and save it to disk"""
        if 'close' not in df.columns:
            raise ValueError("Input DataFrame must contain a 'close' column")
        
        data_hash = str(pd.util.hash_pandas_object(df).sum())
        
        try:
            prepared_data = self._prepare_data(df, context_length, prediction_length)
            total_size = len(prepared_data)
            train_size = int(total_size * train_split)
            val_size = total_size - train_size
            
            save_path = Path(save_path)
            save_path.mkdir(exist_ok=True)
            
            with open(save_path / 'prepared_data.pkl', 'wb') as f:
                pickle.dump(prepared_data, f)
            
            with open(save_path / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.price_scaler, f)
            
            info = PreparedDataInfo(
                context_length=context_length,
                prediction_length=prediction_length,
                train_size=train_size,
                val_size=val_size,
                feature_dim=7,
                data_hash=data_hash,
                scaler_params={
                    'feature_range': self.price_scaler.feature_range,
                    'scale_': self.price_scaler.scale_.tolist(),
                    'min_': self.price_scaler.min_.tolist(),
                },
                preparation_date=datetime.now().isoformat(),
                original_data_shape=df.shape
            )
            
            with open(save_path / 'info.pkl', 'wb') as f:
                pickle.dump(info, f)
            
            return info
            
        except Exception as e:
            logging.error(f"Error during data preparation: {str(e)}")
            raise
    
    def _prepare_data(self, df: pd.DataFrame, context_length: int, prediction_length: int):
        df = df.copy()
        df['scaled_close'] = self.price_scaler.fit_transform(df['close'].values.reshape(-1, 1))
        time_features = self._calculate_time_features(df.index)
        return self._create_sequences(df, time_features, context_length, prediction_length)
    
    def _calculate_time_features(self, time_index):
        minute = time_index.minute
        hour = time_index.hour
        day = time_index.dayofweek
        
        return np.stack([
            np.sin(2 * np.pi * minute / 60),
            np.cos(2 * np.pi * minute / 60),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day / 7),
            np.cos(2 * np.pi * day / 7)
        ], axis=1)
    
    def _create_sequences(self, df, time_features, context_length, prediction_length):
        total_length = len(df)
        num_sequences = total_length - context_length - prediction_length + 1
        
        data = []
        scaled_close = df['scaled_close'].values
        
        for i in range(num_sequences):
            data.append({
                'past_values': scaled_close[i:i+context_length],
                'past_time_features': time_features[i:i+context_length],
                'future_values': scaled_close[i+context_length:i+context_length+prediction_length]
            })
        
        return data

# Training classes
class StockDataset(Dataset):
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

class SimpleStockPredictor(nn.Module):
    def __init__(self, context_length: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=7,  # price + 6 time features
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, past_values, past_time_features):
        # Combine price and time features
        x = torch.cat([
            past_values.unsqueeze(-1),
            past_time_features
        ], dim=-1)
        
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        prediction = self.fc(last_hidden)
        return prediction
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TrainingConfig:
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.num_epochs = kwargs.get('num_epochs', 100)
        self.patience = kwargs.get('patience', 10)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.gradient_clip_val = kwargs.get('gradient_clip_val', 1.0)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = kwargs.get('num_workers', 2 if torch.cuda.is_available() else 0)

class StockTrainer:
    def __init__(self, model: nn.Module, config: TrainingConfig, save_dir: str = 'model_checkpoints'):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.model.to(self.device)
        self.writer = SummaryWriter(log_dir=self.save_dir / 'tensorboard')
        
        num_params = count_parameters(self.model)
        logging.info(f"Initialized trainer with device: {self.device}")
        logging.info(f"Number of trainable parameters: {num_params}")
        print(f"Number of trainable parameters: {num_params}")

    @staticmethod
    def load_prepared_data(data_path: str) -> Tuple[PreparedDataInfo, Any, Any]:
        data_path = Path(data_path)
        with open(data_path / 'info.pkl', 'rb') as f:
            info = pickle.load(f)
        with open(data_path / 'prepared_data.pkl', 'rb') as f:
            prepared_data = pickle.load(f)
        with open(data_path / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return info, prepared_data, scaler
    
    def create_data_loaders(self, prepared_data: list, train_size: int, val_size: int):
        dataset = StockDataset(prepared_data)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )
        
        return train_loader, val_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs} [Train]')
            for past_values, past_time_features, future_values in train_bar:
                past_values = past_values.to(self.device)
                past_time_features = past_time_features.to(self.device)
                future_values = future_values.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(past_values, past_time_features)
                loss = criterion(outputs, future_values[:, 0].unsqueeze(1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_val
                )
                optimizer.step()
                
                train_loss += loss.item()
                train_bar.set_postfix({'loss': f'{loss.item():.8f}'})  # Increased decimal places
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs} [Val]')
            with torch.no_grad():
                for past_values, past_time_features, future_values in val_bar:
                    past_values = past_values.to(self.device)
                    past_time_features = past_time_features.to(self.device)
                    future_values = future_values.to(self.device)
                    
                    outputs = self.model(past_values, past_time_features)
                    loss = criterion(outputs, future_values[:, 0].unsqueeze(1))
                    val_loss += loss.item()
                    val_bar.set_postfix({'loss': f'{loss.item():.8f}'})  # Increased decimal places
            
            val_loss /= len(val_loader)
            
            print(f'Epoch {epoch+1}/{self.config.num_epochs}:')
            print(f'Training Loss: {train_loss:.8f}')  # Increased decimal places
            print(f'Validation Loss: {val_loss:.8f}')  # Increased decimal places
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_dir / 'best_model.pth')
                print(f'New best model saved with validation loss: {val_loss:.8f}')  # Increased decimal places
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break

# Example usage
def train_stock_predictor(df: pd.DataFrame, 
                         context_length: int = 60,
                         prediction_length: int = 30,
                         batch_size: int = 32,
                         num_epochs: int = 100):
    """
    Main function to prepare data and train the model
    
    Args:
        df: DataFrame with 'time' index and 'close' column
        context_length: Number of time steps to use as input
        prediction_length: Number of time steps to predict
        batch_size: Training batch size
        num_epochs: Number of training epochs
    """
    # Prepare data
    data_prep = DataPreparation()
    prepared_info = data_prep.prepare_and_save(
        df=df,
        context_length=context_length,
        prediction_length=prediction_length,
        save_path='prepared_data'
    )
    
    # Load prepared data
    info, prepared_data, scaler = StockTrainer.load_prepared_data('prepared_data')
    
    # Create model and configuration
    model = SimpleStockPredictor(context_length=context_length)
    config = TrainingConfig(batch_size=batch_size, num_epochs=num_epochs)
    
    # Create trainer
    trainer = StockTrainer(model=model, config=config)
    
    # Create data loaders
    train_loader, val_loader = trainer.create_data_loaders(
        prepared_data=prepared_data,
        train_size=info.train_size,
        val_size=info.val_size
    )
    
    # Train the model
    trainer.train(train_loader, val_loader)
    
    return trainer, scaler

# Example usage
"""
# Load your data
df = pd.read_csv('your_data.csv', parse_dates=['time'], index_col='time')

# Train the model
trainer, scaler = train_stock_predictor(
    df=df,
    context_length=60,  # 1 hour if data is minute-by-minute
    prediction_length=30,  # 30 minutes prediction
    batch_size=32,
    num_epochs=100
)
"""




if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('bitcoin_minute_data.csv', parse_dates=['time'], index_col='time')

    # Train the model with longer context length
    trainer, scaler = train_stock_predictor(
        df=df,
        context_length=240,  # 4 hours if data is minute-by-minute
        prediction_length=30,  # 30 minutes prediction
        batch_size=32,
        num_epochs=100
    )

    print("Training completed.")