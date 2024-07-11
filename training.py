# training.py
import torch
from torch.utils.data import DataLoader, Dataset

class CryptoDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_models(ts_model, llama_model, prepared_data):
    dataset = CryptoDataset(prepared_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    ts_optimizer = torch.optim.AdamW(ts_model.parameters(), lr=1e-4)
    llama_optimizer = torch.optim.AdamW(llama_model.parameters(), lr=1e-5)
    
    for epoch in range(10):  # Number of epochs
        for batch in dataloader:
            # Time Series Transformer training
            ts_outputs = ts_model(
                past_values=batch['past_values'],
                past_time_features=batch['past_time_features'],
                future_values=batch['future_values'],
                future_time_features=batch['future_time_features']
            )
            ts_loss = ts_outputs.loss
            ts_loss.backward()
            ts_optimizer.step()
            ts_optimizer.zero_grad()
            
            # Extract features for LLaMA
            with torch.no_grad():
                extracted_features = ts_model(
                    past_values=batch['past_values'],
                    past_time_features=batch['past_time_features']
                ).last_hidden_state
            
            # Prepare input for LLaMA (you might need to adjust this based on your specific requirements)
            llama_input = llama_tokenizer(extracted_features.tolist(), return_tensors="pt", padding=True)
            
            # LLaMA training
            llama_outputs = llama_model(**llama_input, labels=batch['future_values'])
            llama_loss = llama_outputs.loss
            llama_loss.backward()
            llama_optimizer.step()
            llama_optimizer.zero_grad()
        
        print(f"Epoch {epoch+1}, TS Loss: {ts_loss.item()}, LLaMA Loss: {llama_loss.item()}")

    torch.save(ts_model.state_dict(), "ts_model.pth")
    torch.save(llama_model.state_dict(), "llama_model.pth")

# Usage
from data_preparation import prepare_data_for_model, fetch_data_from_influx
from model_setup import setup_time_series_transformer, setup_llama

df = fetch_data_from_influx("crypto_bucket", "BTC_price", "-30d", "now()")
prepared_data = prepare_data_for_model(df, context_length=50, prediction_length=10)

ts_model = setup_time_series_transformer()
llama_model, _ = setup_llama()

train_models(ts_model, llama_model, prepared_data)
