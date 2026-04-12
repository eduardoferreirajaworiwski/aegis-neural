# Aegis-Neural

Aegis-Neural is an unsupervised anomaly detection system for cybersecurity logs, utilizing an **Autoencoder** architecture to identify patterns and deviations in network traffic and system events.

## Technical Overview
- **Architecture:** Deep Autoencoder for dimensionality reduction and reconstruction error analysis.
- **Goal:** Detect anomalies in high-dimensional log data without labeled training sets.
- **Framework:** PyTorch.

## Core Features
- **Unsupervised Learning:** Learns the "normal" state of system logs.
- **Anomaly Scoring:** Uses Reconstruction Error (MSE) to flag suspicious activity.
- **Explainable Layers:** Documented tensor transformations for every layer.
