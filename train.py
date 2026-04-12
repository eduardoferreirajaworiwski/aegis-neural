import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model import AegisAutoencoder
from data_engine import generate_normal_logs, preprocess_and_convert_to_tensor

def simulate_anomaly(model, preprocessor):
    print("\n--- Anomaly Simulation ---")
    
    # 1. Normal Log
    normal_df = pd.DataFrame({
        'duration': [5.0],
        'bytes_sent': [2500.0],
        'protocol': ['TCP'],
        'status': ['SUCCESS']
    })
    
    # 2. Malicious Log (Data Exfiltration: 100x bytes_sent)
    malicious_df = pd.DataFrame({
        'duration': [5.0],
        'bytes_sent': [250000.0], # 100x larger
        'protocol': ['TCP'],
        'status': ['SUCCESS']
    })
    
    # Preprocess using the fitted scaler/encoder
    normal_tensor = torch.tensor(preprocessor.transform(normal_df), dtype=torch.float32)
    malicious_tensor = torch.tensor(preprocessor.transform(malicious_df), dtype=torch.float32)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        normal_reconstruction = model(normal_tensor)
        malicious_reconstruction = model(malicious_tensor)
        
        normal_mse = nn.MSELoss()(normal_reconstruction, normal_tensor).item()
        malicious_mse = nn.MSELoss()(malicious_reconstruction, malicious_tensor).item()
        
    print(f"Normal Log Reconstruction Error (MSE): {normal_mse:.4f}")
    print(f"Malicious Log Reconstruction Error (MSE): {malicious_mse:.4f}")
    print(f"Anomaly Multiplier: {malicious_mse / normal_mse:.2f}x higher error")

if __name__ == '__main__':
    # 1. Load Data
    print("Loading and preprocessing data...")
    df = generate_normal_logs(n_samples=1000)
    tensor_data, preprocessor = preprocess_and_convert_to_tensor(df)
    
    # Validation Split (80/20) - Rule: Every training script must include a validation step
    train_size = int(0.8 * len(tensor_data))
    train_data = tensor_data[:train_size]
    val_data = tensor_data[train_size:]
    
    # 2. Model, Loss, Optimizer
    model = AegisAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    epochs = 50
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        outputs = model(train_data)
        loss = criterion(outputs, train_data)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation - Rule: Every training script must include a validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data)
            val_loss = criterion(val_outputs, val_data)
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            
    # 4. Anomaly Simulation
    simulate_anomaly(model, preprocessor)
