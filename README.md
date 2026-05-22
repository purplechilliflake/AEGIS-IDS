# AEGIS — ML-Driven Intrusion Detection System

> Distributed intrusion detection using ensemble machine learning across multiple network security datasets, with a blockchain-backed immutable alert log.

---

## Overview

AEGIS detects anomalies in network traffic in real time by running trained Random Forest classifiers against live data streams. Detected intrusions are logged as tamper-resistant entries — either via a pure-Python blockchain simulation or a Solidity smart contract deployed on a local Ethereum chain (Ganache).

The system was trained and evaluated across four public network security datasets, achieving **99.9% detection accuracy** on the NSL-KDD test split.

---

## Architecture

```
Data Sources (NSL-KDD · UNSW-NB15 · CIC-IDS 2017 · BoT-IoT)
        │
        ▼
Preprocessing (label encoding · normalization · SMOTE · feature engineering)
        │
        ▼
ML Training (RandomForestClassifier · n=200 · depth=15 · 5-fold CV)
        │
        ▼
Real-time Detection (streaming predict → anomaly flag)
        │
        ▼
Blockchain Logging (SHA-256 chain · AlertLogger.sol · Web3 / Ganache)
```

---

## Features

- **Multi-dataset support** — separate preprocessing pipelines for NSL-KDD, UNSW-NB15, CIC-IDS 2017, and BoT-IoT
- **Feature engineering** — derived features (`bytes_rate`, `packet_ratio`) and attack family categorization (DoS, Probe, R2L, U2R)
- **Class imbalance handling** — SMOTE oversampling for IoT traffic datasets
- **Dual blockchain logging**:
  - Pure-Python blockchain simulation with SHA-256 hashing, chain validation, and node reputation scoring
  - Solidity smart contract (`AlertLogger.sol`) deployed on Ganache with Web3.py transaction signing
- **Evaluation suite** — confusion matrix, ROC curve, AUC, and feature importance plots

---

## Results

| Dataset     | Model              | Accuracy  |
|-------------|--------------------|-----------|
| NSL-KDD     | Random Forest      | **99.9%** |
| BoT-IoT     | Random Forest + SMOTE | ~99%   |
| UNSW-NB15   | Random Forest      | —         |
| CIC-IDS 2017| Random Forest      | —         |

---

## Project Structure

```
AEGIS/
├── src/
│   ├── 1_data_preprocessing.py       # NSL-KDD preprocessing pipeline
│   ├── 2_model_training.py           # Model training + evaluation
│   ├── 3_model_evaluation.py         # Visualizations (ROC, confusion matrix)
│   ├── 4_compare_baselines.py        # Baseline comparisons
│   ├── 5_full_pipeline.py            # End-to-end pipeline runner
│   ├── blockchain_ids_simulation.py  # Pure-Python blockchain simulation
│   └── BoT-IoT/                      # BoT-IoT specific scripts
├── contracts/
│   └── AlertLogger.sol               # Solidity alert logging contract
├── ml_to_blockchain_logger.py        # UNSW-NB15 → blockchain logger
├── ml_to_blockchain_logger_cicids.py # CIC-IDS → blockchain logger
├── deploy_alert_logger.py            # Contract deployment script
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.9+
- Node.js (for Truffle/Ganache)
- [Ganache](https://trufflesuite.com/ganache/) — local Ethereum chain

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download datasets

The datasets are not included in this repo due to size. Download them and place under `data/`:

| Dataset | Link |
|---------|------|
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| CIC-IDS 2017 | https://www.unb.ca/cic/datasets/ids-2017.html |
| BoT-IoT | https://research.unsw.edu.au/projects/bot-iot-dataset |

Expected structure:
```
data/
├── NSL-KDD/
│   ├── KDDTrain+.csv
│   └── KDDTest+.csv
├── unsw-nb15/
├── CIC-IDS2017/
└── BoT-IoT/
```

### Environment variables

```bash
cp .env.example .env
# Edit .env and add your Ganache private key
```

---

## Usage

### 1. Preprocess data

```bash
python src/1_data_preprocessing.py
```

### 2. Train the model

```bash
python src/2_model_training.py
```

### 3. Evaluate

```bash
python src/3_model_evaluation.py
# Results saved to results/
```

### 4. Run blockchain simulation

```bash
python src/blockchain_ids_simulation.py
```

### 5. Log alerts to smart contract (requires Ganache running)

```bash
# Deploy contract first
python deploy_alert_logger.py

# Run the logger
python ml_to_blockchain_logger.py
```

---

## Blockchain Component

Two approaches are implemented:

**Simulation** (`blockchain_ids_simulation.py`) — a self-contained Python implementation with SHA-256 block hashing, chain integrity validation, and a node reputation system that adjusts trust scores based on detection confidence.

**On-chain** (`AlertLogger.sol`) — a Solidity contract that stores alerts (timestamp, type, source IP, destination IP) on a local Ganache chain. Each detection event is signed and submitted as a transaction, producing an immutable, verifiable audit trail.

---

## Tech Stack

`Python` `Scikit-learn` `Pandas` `NumPy` `Solidity` `Web3.py` `Ganache` `Matplotlib` `Seaborn` `imbalanced-learn`

---

## License

MIT