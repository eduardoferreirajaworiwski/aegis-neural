import torch
import torch.nn as nn

class AegisAutoencoder(nn.Module):
    """
    Aegis-Neural Autoencoder for Anomaly Detection.

    LATENT SPACE EXPLANATION:
    The Bottleneck (2 neurons) represents the 'Latent Space'—a highly compressed
    representation of the essential features of 'Normal' network logs. 
    
    WHY RECONSTRUCTION ERROR (MSE) DETECTS ANOMALIES:
    Since the network is only trained to compress and decompress 'Normal' traffic 
    patterns through this narrow bottleneck, it becomes very good at reconstructing them. 
    However, when an anomaly (e.g., a cyberattack with unusual ports or packet sizes) 
    is passed through the network, the bottleneck won't know how to represent these 
    novel, malicious patterns. Consequently, the Decoder will fail to reconstruct 
    the original input accurately. This will result in a high Mean Squared Error (MSE)
    between the input and output, flagging the event as an anomaly.
    """
    
    def __init__(self):
        super(AegisAutoencoder, self).__init__()
        
        # --- ENCODER SECTION ---
        # Layer 1: Compress 7 input features to 4 neurons
        # Shape Transformation: [Batch_Size, 7] -> [Batch_Size, 4]
        self.encoder_layer1 = nn.Linear(7, 4)
        self.relu1 = nn.ReLU()
        
        # Layer 2 (Bottleneck): Compress 4 neurons to 2 neurons (Latent Space)
        # Shape Transformation: [Batch_Size, 4] -> [Batch_Size, 2]
        self.encoder_bottleneck = nn.Linear(4, 2)
        self.relu2 = nn.ReLU()
        
        # --- DECODER SECTION ---
        # Layer 3: Decompress 2 neurons (Latent Space) back to 4 neurons
        # Shape Transformation: [Batch_Size, 2] -> [Batch_Size, 4]
        self.decoder_layer1 = nn.Linear(2, 4)
        self.relu3 = nn.ReLU()
        
        # Layer 4 (Output): Reconstruct the original 7 features from 4 neurons
        # Shape Transformation: [Batch_Size, 4] -> [Batch_Size, 7]
        # Note: No activation here because inputs are Standard Scaled (can be negative/positive)
        self.decoder_output = nn.Linear(4, 7)

    def forward(self, x):
        # Forward pass through Encoder
        x = self.encoder_layer1(x)
        x = self.relu1(x)
        
        x = self.encoder_bottleneck(x)
        x = self.relu2(x)
        
        # Forward pass through Decoder
        x = self.decoder_layer1(x)
        x = self.relu3(x)
        
        x = self.decoder_output(x)
        return x

if __name__ == '__main__':
    # Instantiate the model
    model = AegisAutoencoder()
    print("AegisAutoencoder Architecture:")
    print(model)
    
    # Calculate trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {total_params}")
