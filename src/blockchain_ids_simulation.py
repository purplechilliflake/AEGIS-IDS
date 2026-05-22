import hashlib
import json
import time
from typing import List, Dict

class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data  # list of transactions
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict] = []
        self.reputation: Dict[str, float] = {}
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, time.time(), [], "0")
        self.chain.append(genesis_block)

    def add_transaction(self, attack_type: str, confidence: float, source_node: str):
        tx = {
            "attack_type": attack_type,
            "confidence": confidence,
            "source_node": source_node
        }
        self.pending_transactions.append(tx)

        # Adjust reputation based on confidence
        if source_node not in self.reputation:
            self.reputation[source_node] = 1.0
        adjustment = 0.1 if confidence > 0.9 else -0.1
        self.reputation[source_node] = max(0.0, min(1.0, self.reputation[source_node] + adjustment))

    def mine_block(self):
        if not self.pending_transactions:
            print("🚫 No transactions to mine.")
            return

        last_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            data=self.pending_transactions.copy(),
            previous_hash=last_block.hash
        )
        self.chain.append(new_block)
        self.pending_transactions = []
        print(f"⛏️  Mined block #{new_block.index} with {len(new_block.data)} transactions.")

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current, previous = self.chain[i], self.chain[i - 1]
            if current.hash != current.calculate_hash():
                return False
            if current.previous_hash != previous.hash:
                return False
        return True

    def print_chain(self):
        print("\n📜 Blockchain:")
        for block in self.chain:
            print(json.dumps({
                'index': block.index,
                'timestamp': time.ctime(block.timestamp),
                'data': block.data,
                'hash': block.hash,
                'previous_hash': block.previous_hash
            }, indent=2))

    def print_reputation(self):
        print("\n📊 Reputation Scores:")
        for node, score in self.reputation.items():
            print(f"Node {node}: {score:.2f}")

# 🧪 Sample Usage
if __name__ == "__main__":
    bc = Blockchain()

    # Simulate detection results
    bc.add_transaction("DoS", 0.97, "node-A")
    bc.add_transaction("Probe", 0.65, "node-B")
    bc.add_transaction("R2L", 0.93, "node-A")

    bc.mine_block()

    bc.add_transaction("U2R", 0.88, "node-C")
    bc.add_transaction("DoS", 0.99, "node-A")

    bc.mine_block()

    bc.print_chain()
    bc.print_reputation()

    print("\n✅ Blockchain valid:", bc.is_chain_valid())
