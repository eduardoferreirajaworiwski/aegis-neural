import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def generate_normal_logs(n_samples=1000):
    """
    Generates a synthetic dataset of 'Normal' network logs.
    """
    np.random.seed(42) # For reproducibility
    
    # Generate numerical features
    duration = np.random.uniform(0.1, 10.0, n_samples)
    bytes_sent = np.random.uniform(100, 5000, n_samples)
    
    # Generate categorical features
    protocols = np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples)
    statuses = np.random.choice(['SUCCESS', 'LOGIN'], n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'duration': duration,
        'bytes_sent': bytes_sent,
        'protocol': protocols,
        'status': statuses
    })
    
    return df

def preprocess_and_convert_to_tensor(df):
    """
    Preprocesses the DataFrame and converts it to a PyTorch Tensor.
    
    WHY DO WE NEED TO NORMALIZE (SCALE) DATA FOR NEURAL NETWORKS?
    Neural networks use gradient descent to optimize their weights. If features
    are on vastly different scales (e.g., duration is 0.1-10.0, bytes_sent is 100-5000),
    the gradients will be disproportionate. The larger feature (bytes_sent) will dominate the
    learning process, leading to slow or unstable convergence. By scaling all
    numerical features to have a mean of 0 and standard deviation of 1 (StandardScaler),
    we ensure all features contribute proportionally, allowing the neural network to
    learn efficiently and stably.
    """
    
    numeric_features = ['duration', 'bytes_sent']
    categorical_features = ['protocol', 'status']
    
    # Define preprocessing pipeline
    # sparse_output=False ensures a dense array is returned, which is required for PyTorch conversion
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ])
    
    # Apply preprocessing
    processed_data = preprocessor.fit_transform(df)
    
    # Convert to PyTorch Tensor
    # processed_data is cast to float32, the standard dtype for PyTorch neural network weights
    tensor_data = torch.tensor(processed_data, dtype=torch.float32)
    
    return tensor_data, preprocessor

if __name__ == "__main__":
    df = generate_normal_logs()
    print(f"Generated DataFrame shape: {df.shape}")
    
    tensor_data, preprocessor = preprocess_and_convert_to_tensor(df)
    print(f"Converted Tensor shape: {tensor_data.shape}")
    print(f"Converted Tensor dtype: {tensor_data.dtype}")
