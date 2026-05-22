from web3 import Web3
import json

# === Load the ABI ===
with open("compiled_alert_logger.json") as f:
    compiled = json.load(f)

abi = compiled["contracts"]["AlertLogger.sol"]["AlertLogger"]["abi"]
contract_address = "0x564F89276DD753F4C1A7ebF3972b834ec6413647"  # replace this

# === Connect to Ganache ===
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))

# === Use your address and private key ===
my_address = w3.eth.accounts[0]  # or paste the same address used before
private_key = "0x785cec6b124672f1c48f014578b043734bd2afd41424f2dd230472c4e1798330"  # same one from deploy script

# === Get Contract ===
contract = w3.eth.contract(address=contract_address, abi=abi)

# === Example Alert Details ===
timestamp = "2025-05-16 01:45:00"
alert_type = "AnomalyDetected"
source_ip = "192.168.1.5"
dest_ip = "10.0.0.20"

# === Build Transaction ===
nonce = w3.eth.get_transaction_count(my_address)
txn = contract.functions.logAlert(timestamp, alert_type, source_ip, dest_ip).build_transaction({
    'chainId': 1337,
    'gas': 500000,
    'gasPrice': w3.to_wei("20", "gwei"),
    'from': my_address,
    'nonce': nonce,
})

# === Sign and Send ===
signed_txn = w3.eth.account.sign_transaction(txn, private_key)
tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print("✅ Alert logged successfully!")
print("🔗 Transaction Hash:", receipt.transactionHash.hex())
