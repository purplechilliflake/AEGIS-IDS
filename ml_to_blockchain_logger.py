from web3 import Web3
import json
import time
from datetime import datetime
import joblib
import pandas as pd

# === Load ABI and contract ===
with open("compiled_alert_logger.json") as f:
    compiled = json.load(f)

abi = compiled["contracts"]["AlertLogger.sol"]["AlertLogger"]["abi"]
contract_address = "0x564F89276DD753F4C1A7ebF3972b834ec6413647"
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
my_address = w3.eth.accounts[0]
private_key = "0x785cec6b124672f1c48f014578b043734bd2afd41424f2dd230472c4e1798330"
contract = w3.eth.contract(address=contract_address, abi=abi)

# === Load model and expected features
model = joblib.load("models/intrusion_detection_rf.pkl")
expected_features = model.feature_names_in_

# === Load test data
all_data = pd.read_csv("data/unsw-nb15/unsw_nb15_processed.csv")

# Drop label column if exists
if 'label' in all_data.columns:
    all_data = all_data.drop(columns=['label'])

# === Feature engineering to match training phase
# Add bytes_rate and packet_ratio
all_data["bytes_rate"] = all_data["dst_bytes"] / (all_data["duration"] + 1)
all_data["packet_ratio"] = all_data["count"] / (all_data["srv_count"] + 1)

# Add placeholder attack_category (0 if not used, or reconstruct it from original label if known)
all_data["attack_category"] = 0

# === Ensure all expected columns are present
missing_cols = set(expected_features) - set(all_data.columns)
for col in missing_cols:
    all_data[col] = 0  # safe default

# Align with model input
incoming_data = all_data[expected_features]

# === Detect anomalies
def detect_anomalies():
    alerts = []
    predictions = model.predict(incoming_data)
    for i, pred in enumerate(predictions):
        if pred == 1:  # 1 = anomaly
            alerts.append({
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "alert_type": "Anomaly",
                "source_ip": f"192.168.1.{100 + i}",
                "dest_ip": f"10.0.0.{50 + i}"
            })
    return alerts

# === Log to blockchain
def log_alert_to_blockchain(alert):
    nonce = w3.eth.get_transaction_count(my_address)
    txn = contract.functions.logAlert(
        alert["timestamp"],
        alert["alert_type"],
        alert["source_ip"],
        alert["dest_ip"]
    ).build_transaction({
        'chainId': 1337,
        'gas': 500000,
        'gasPrice': w3.to_wei("20", "gwei"),
        'from': my_address,
        'nonce': nonce,
    })

    signed_txn = w3.eth.account.sign_transaction(txn, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    print(f"✅ Logged alert: {alert['alert_type']} from {alert['source_ip']}")
    print(f"🔗 Transaction Hash: {receipt.transactionHash.hex()}\n")

# === Run
if __name__ == "__main__":
    alerts = detect_anomalies()
    for alert in alerts:
        log_alert_to_blockchain(alert)
        time.sleep(2)
