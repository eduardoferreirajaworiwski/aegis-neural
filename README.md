# Aegis-Neural: Unsupervised Anomaly Detection

Aegis-Neural is a cybersecurity-focused anomaly detection system that leverages a **Deep Autoencoder** architecture to identify suspicious patterns in network logs. By learning a compressed representation of "normal" system behavior, it can flag anomalies—such as data exfiltration or unauthorized access—based on their reconstruction error.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.12+
- Virtual Environment support

### 2. Installation
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Usage
Run the training pipeline to generate synthetic data, train the model, and simulate an anomaly:
```bash
python3 train.py
```

## 🧠 Architecture Overview

The system uses a symmetric Autoencoder (7-4-2-4-7) implemented in PyTorch:

- **Input Layer (7):** Processed log features (duration, bytes, protocol, status).
- **Encoder:** Compresses features into a lower-dimensional representation.
- **Bottleneck (2):** The **Latent Space**. This represents the most critical "DNA" of normal traffic.
- **Decoder:** Attempts to reconstruct the original 7 features from the latent space.
- **Output Layer (7):** The reconstructed log entry.

### Why Reconstruction Error?
In an unsupervised setting, the model only sees "Normal" data during training. It learns to compress and decompress these patterns with high precision. When a **Malicious Log** (e.g., unusual port or massive data transfer) passes through, the model lacks the "knowledge" to compress it effectively, resulting in a high **Mean Squared Error (MSE)**. This error serves as our anomaly score.

## 🛠 Project Structure

- `data_engine.py`: Synthetic log generation and `scikit-learn` preprocessing pipeline.
- `model.py`: PyTorch implementation of the `AegisAutoencoder`.
- `train.py`: Training loop with validation split and anomaly simulation.
- `GEMINI.md`: Real-time operational dashboard for tracking model topology and loss.
- `gemini_rules.md`: Project-specific coding standards and safety mandates.

## 📊 Dashboard
Track the project's health and training progress in [GEMINI.md](./GEMINI.md).
