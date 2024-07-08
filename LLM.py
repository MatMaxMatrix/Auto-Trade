from huggingface_hub import hf_hub_download
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import TimeSeriesTransformerForPrediction, AdamW
import matplotlib.pyplot as plt

# Load the data
file = hf_hub_download(
    repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

# Create a dataset
dataset = TensorDataset(
    batch["past_values"],
    batch["past_time_features"],
    batch["past_observed_mask"],
    batch["static_categorical_features"],
    batch["static_real_features"],
    batch["future_values"],
    batch["future_time_features"]
)

# Split dataset into train and validation
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# Initialize the model
model = TimeSeriesTransformerForPrediction.from_pretrained(
    "huggingface/time-series-transformer-tourism-monthly"
)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Set up scheduler
num_epochs = 1000
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_val_loss = float('inf')
patience = 100
counter = 0

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Move batch to device
        batch = [b.to(device) for b in batch]
        
        outputs = model(
            past_values=batch[0],
            past_time_features=batch[1],
            past_observed_mask=batch[2],
            static_categorical_features=batch[3],
            static_real_features=batch[4],
            future_values=batch[5],
            future_time_features=batch[6]
        )
        
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = [b.to(device) for b in batch]
            outputs = model(
                past_values=batch[0],
                past_time_features=batch[1],
                past_observed_mask=batch[2],
                static_categorical_features=batch[3],
                static_real_features=batch[4],
                future_values=batch[5],
                future_time_features=batch[6]
            )
            val_loss += outputs.loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Learning rate scheduling
    scheduler.step()
    
    # # Early stopping
    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     counter = 0
    #     torch.save(model.state_dict(), 'best_model.pth')
    # else:
    #     counter += 1
    #     if counter >= patience:
    #         print("Early stopping")
    #         break

# Save the final trained model
model.save_pretrained("trained_time_series_model")
print("Training completed!")

# Plot training and validation loss
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()

# Print final learning rate
print(f"Final learning rate: {optimizer.param_groups[0]['lr']}")

# Monitor gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")



