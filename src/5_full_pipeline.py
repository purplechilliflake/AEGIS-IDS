import pandas as pd
import joblib
from blockchain_ids_simulation import Blockchain

def run_pipeline():
    # 1. Load model
    model = joblib.load("models/intrusion_detection_rf.pkl")

    # 2. Load test data
    test = pd.read_csv("data/processed/test.csv")
    drop_cols = [col for col in ['class', 'label', 'difficulty_level'] if col in test.columns]
    X_test = test.drop(columns=drop_cols, errors='ignore')
    y_test = test['label']

    # 3. Make predictions
    y_pred = model.predict(X_test)

    # 4. Initialize blockchain
    blockchain = Blockchain()

    # 5. Log first 10 anomaly detections to blockchain
    for i in range(min(10, len(y_pred))):
        if y_pred[i] == 1:  # Only log anomalies
            blockchain.add_transaction(
                attack_type="Anomaly", 
                confidence=0.95,  # Placeholder, could be replaced by predicted probability
                source_node=f"node-{i % 3}"  # Simulate nodes rotating: node-0, node-1, node-2
            )

    # 6. Mine the block
    blockchain.mine_block()

    # 7. Output results
    blockchain.print_chain()
    blockchain.print_reputation()
    print("\n✅ Pipeline complete. Blockchain length:", len(blockchain.chain))

if __name__ == "__main__":
    run_pipeline()
