from solcx import compile_standard, install_solc
import json
from web3 import Web3

# Step 1: Install Solidity compiler
install_solc("0.8.0")

# Step 2: Load the Solidity source code
with open("contracts/AlertLogger.sol", "r") as file:
    alert_logger_source = file.read()

# Step 3: Compile the contract
compiled_sol = compile_standard(
    {
        "language": "Solidity",
        "sources": {
            "AlertLogger.sol": {
                "content": alert_logger_source
            }
        },
        "settings": {
            "outputSelection": {
                "*": {
                    "*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]
                }
            }
        },
    },
    solc_version="0.8.0",
)

# Step 4: Save compiled output
with open("compiled_alert_logger.json", "w") as file:
    json.dump(compiled_sol, file)

# Extract contract data
bytecode = compiled_sol["contracts"]["AlertLogger.sol"]["AlertLogger"]["evm"]["bytecode"]["object"]
abi = compiled_sol["contracts"]["AlertLogger.sol"]["AlertLogger"]["abi"]

# Step 5: Connect to Ganache
ganache_url = "http://127.0.0.1:7545"  # Make sure Ganache is running!
w3 = Web3(Web3.HTTPProvider(ganache_url))

chain_id = 1337
my_address = w3.eth.accounts[0]
private_key = "0x785cec6b124672f1c48f014578b043734bd2afd41424f2dd230472c4e1798330"  # You will replace this manually

# Step 6: Create the contract instance
AlertLogger = w3.eth.contract(abi=abi, bytecode=bytecode)

# Step 7: Build and sign the transaction
nonce = w3.eth.get_transaction_count(my_address)
transaction = AlertLogger.constructor().build_transaction({
    "chainId": chain_id,
    "gas": 5000000,
    "gasPrice": w3.to_wei("20", "gwei"),
    "from": my_address,
    "nonce": nonce,
})

# Sign and send the transaction
signed_txn = w3.eth.account.sign_transaction(transaction, private_key=private_key)
tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)

# Step 8: Wait for confirmation
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
print("✅ Contract deployed at:", tx_receipt.contractAddress)
