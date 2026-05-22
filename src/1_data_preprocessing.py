import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import numpy as np
import json

class NslKddPreprocessor:
    """
    Handles comprehensive preprocessing of NSL-KDD dataset
    including feature engineering and attack categorization
    """
    
    def __init__(self):
        self.encoders = {}
        self.scaler = None
        self.feature_names = None
        
    def _load_data(self, path, is_train=False):
        """Load raw data with proper column names"""
        cols = [
            'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
            'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
            'root_shell','su_attempted','num_root','num_file_creations','num_shells',
            'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
            'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
            'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
            'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
            'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
            'class', 'difficulty_level'  # Added difficulty_level as it might be in the test dataset
        ]
        
        # Load the data
        df = pd.read_csv(path, header=None if not is_train else 0)
        
        # Check the number of columns and assign only the necessary column names
        actual_cols = cols[:df.shape[1]]
        df.columns = actual_cols
        
        # If we're missing expected columns, add them with NaN values
        for col in cols[:41]:  # Excluding 'class' and 'difficulty_level'
            if col not in df.columns:
                df[col] = np.nan
                
        return df
    
    def _engineer_features(self, df):
        """Create new derived features"""
        # Interaction features
        df['bytes_rate'] = (df['src_bytes'] + df['dst_bytes']) / (df['duration'] + 1e-6)
        df['packet_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1e-6)
        
        # Attack category mapping
        attack_map = {
            'normal': 'normal',
            'neptune': 'dos', 'smurf': 'dos', 'back': 'dos', 'teardrop': 'dos',
            'pod': 'dos', 'land': 'dos', 'apache2': 'dos', 'udpstorm': 'dos',
            'processtable': 'dos', 'mailbomb': 'dos',
            'portsweep': 'probe', 'ipsweep': 'probe', 'nmap': 'probe', 'satan': 'probe',
            'mscan': 'probe', 'saint': 'probe',
            'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l', 'phf': 'r2l',
            'multihop': 'r2l', 'warezmaster': 'r2l', 'warezclient': 'r2l',
            'spy': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l', 'snmpguess': 'r2l',
            'snmpgetattack': 'r2l', 'httptunnel': 'r2l', 'sendmail': 'r2l',
            'named': 'r2l', 'worm': 'r2l',
            'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r', 
            'perl': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r'
        }
        
        # Handle cases where attacks in the test set might not be in the mapping
        df['attack_category'] = df['class'].apply(lambda x: attack_map.get(x.lower() if isinstance(x, str) else x, 'unknown'))
        return df
    
    def preprocess(self, train_path, test_path):
        """Main preprocessing pipeline"""
        # Load datasets
        print(f"Loading training data from {train_path}")
        train_df = self._load_data(train_path, is_train=True)
        print(f"Training data shape: {train_df.shape}")
        
        print(f"Loading test data from {test_path}")
        test_df = self._load_data(test_path)
        print(f"Test data shape: {test_df.shape}")
        
        # Feature engineering
        print("Performing feature engineering...")
        train_df = self._engineer_features(train_df)
        test_df = self._engineer_features(test_df)
        
        # Label encoding
        print("Encoding labels...")
        train_df['label'] = train_df['class'].apply(lambda x: 0 if isinstance(x, str) and 'normal' in x.lower() else 1)
        test_df['label'] = test_df['class'].apply(lambda x: 0 if isinstance(x, str) and 'normal' in x.lower() else 1)
        
        # Encode categoricals
        print("Encoding categorical features...")
        cat_cols = ['protocol_type', 'service', 'flag', 'attack_category']
        for col in cat_cols:
            le = LabelEncoder()
            combined_values = pd.concat([train_df[col], test_df[col]]).unique()
            le.fit(combined_values)  # Fit on combined unique values to handle all possible values
            train_df[col] = le.transform(train_df[col])
            test_df[col] = le.transform(test_df[col])
            self.encoders[col] = le
            print(f"  Encoded {col} with {len(le.classes_)} unique values")
        
        # Normalize numerical features
        print("Normalizing numerical features...")
        num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
        num_cols = [c for c in num_cols if c not in cat_cols + ['label', 'class', 'difficulty_level']]
        
        self.scaler = MinMaxScaler()
        train_df[num_cols] = self.scaler.fit_transform(train_df[num_cols])
        test_df[num_cols] = self.scaler.transform(test_df[num_cols])
        
        # Split train into train/val
        print("Splitting training data into train/validation sets...")
        train, val = train_test_split(
            train_df, test_size=0.2, stratify=train_df['label'], random_state=42
        )
        
        # Save metadata
        self.feature_names = list(train.drop(['class', 'label', 'difficulty_level'] 
                                            if 'difficulty_level' in train.columns 
                                            else ['class', 'label'], axis=1).columns)
        self._save_metadata()
        
        return train, val, test_df
    
    def _save_metadata(self):
        """Save encoders and feature info"""
        os.makedirs("models", exist_ok=True)
        metadata = {
            'feature_names': self.feature_names,
            'categorical_mappings': {
                col: {str(k): int(v) for k, v in 
                     {c: i for i, c in enumerate(le.classes_)}.items()}
                for col, le in self.encoders.items()
            }
        }
        with open("models/preprocessor_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved preprocessor metadata to models/preprocessor_metadata.json")

if __name__ == "__main__":
    print("Starting NSL-KDD preprocessing...")
    preprocessor = NslKddPreprocessor()
    
    train_path = "data/NSL-KDD/KDDTrain+.csv"
    test_path = "data/NSL-KDD/KDDTest+.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"❌ Error: Data files not found.")
        print(f"Please ensure that the following files exist:")
        print(f"- {train_path}")
        print(f"- {test_path}")
        exit(1)
    
    train, val, test = preprocessor.preprocess(train_path, test_path)
    
    # Save processed data
    print("Saving processed datasets...")
    os.makedirs("data/processed", exist_ok=True)
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    
    print("\n✅ Preprocessing complete!")
    print(f"Train: {len(train)} samples")
    print(f"Validation: {len(val)} samples") 
    print(f"Test: {len(test)} samples")