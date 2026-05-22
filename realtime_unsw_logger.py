from web3 import Web3
import pandas as pd
import joblib
import time
from datetime import datetime

# === Blockchain Setup ===
with open("compiled_alert_logger.json") as f:
    import json
    compiled = json.load(f)

abi = compiled["contracts"]["AlertLogger.sol"]["AlertLogger"]["abi"]
contract_address = "0x564F89276DD753F4C1A7ebF3972b834ec6413647"

w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
my_address = w3.eth.accounts[0]
private_key = "0x785cec6b124672f1c48f014578b043734bd2afd41424f2dd230472c4e1798330"
contract = w3.eth.contract(address=contract_address, abi=abi)

# === Load Refined Model + Feature Columns
model = joblib.load("models/unsw_rf_refined.pkl")
expected_features = joblib.load("models/unsw_rf_refined_features.pkl")

# === Load Preprocessed Data (for simulation)
df = pd.read_csv("data/UNSW-NB15/unsw_nb15_processed.csv")
df = df[expected_features + ['label']]  # Ensure correct order

# === Blockchain Logger Function
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

    print(f"✅ Realtime alert logged: {alert['alert_type']} from {alert['source_ip']}")
    print(f"🔗 Tx Hash: {receipt.transactionHash.hex()}\n")

# === Stream and Detect
if __name__ == "__main__":
    print("🚨 Starting real-time intrusion detection (UNSW-NB15)...\n")
    for i, row in df.iterrows():
        X_row = row[expected_features].values.reshape(1, -1)
        prediction = model.predict(X_row)[0]

        if prediction == 1:  # anomaly
            alert = {
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "alert_type": "Anomaly",
                "source_ip": f"10.0.0.{i % 255}",
                "dest_ip": f"172.16.0.{(i * 2) % 255}"
            }
            log_alert_to_blockchain(alert)

        time.sleep(0.75)  # simulate ~1 record per second (adjust as needed)
