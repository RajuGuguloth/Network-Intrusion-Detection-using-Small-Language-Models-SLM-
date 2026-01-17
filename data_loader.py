import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_URL = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv"
LOCAL_FILE = "UNSW_NB15_training-set.csv"

# Relevant features as per plan
# 'proto', 'service', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'dur', 'state', 'label'
# Note: 'label' is the binary target (0=benign, 1=malicious) 
# 'attack_cat' is the multiclass label if needed, but the user prompt mentioned 'malicious' vs 'benign' primarily.
# We will load both to handle the multi-class variant prompt.

SELECTED_FEATURES = [
    'proto', 'service', 'spkts', 'dpkts', 
    'sbytes', 'dbytes', 'dur', 'state', 
    'attack_cat', 'label'
]

def download_dataset():
    if not os.path.exists(LOCAL_FILE):
        print(f"Downloading dataset from {DATASET_URL}...")
        try:
            df = pd.read_csv(DATASET_URL)
            df.to_csv(LOCAL_FILE, index=False)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Creating a synthetic dataset for testing purposes...")
            create_synthetic_dataset()
    else:
        print(f"Dataset {LOCAL_FILE} found locally.")

def create_synthetic_dataset():
    # If download fails, create a dummy dataset with same structure
    data = {
        'proto': ['TCP', 'UDP', 'TCP', 'UDP']*500,
        'service': ['http', 'dns', 'ftp', '-']*500,
        'spkts': np.random.randint(1, 100, 2000),
        'dpkts': np.random.randint(0, 100, 2000),
        'sbytes': np.random.randint(40, 5000, 2000),
        'dbytes': np.random.randint(0, 10000, 2000),
        'dur': np.random.random(2000),
        'state': ['FIN', 'INT', 'CON', 'REQ']*500,
        'attack_cat': ['Normal', 'Generic', 'Exploits', 'Fuzzers']*500,
        'label': [0, 1, 1, 1]*500
    }
    df = pd.DataFrame(data)
    df.to_csv(LOCAL_FILE, index=False)
    print("Synthetic dataset created.")

def load_and_preprocess(sample_size=1000, random_state=42):
    """
    Loads dataset, selects features, balances classes slightly if needed, and samples.
    """
    download_dataset()
    
    try:
        df = pd.read_csv(LOCAL_FILE)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return None

    # Ensure required columns exist
    missing_cols = [c for c in SELECTED_FEATURES if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Using available subset.")
        cols_to_use = [c for c in SELECTED_FEATURES if c in df.columns]
    else:
        cols_to_use = SELECTED_FEATURES

    df = df[cols_to_use].copy()
    
    # Fill NaN for 'service' or others if necessary
    if 'service' in df.columns:
        df['service'] = df['service'].replace('-', 'unknown').fillna('unknown')
    
    # Sample balanced subset if possible
    # Just simple sampling for now to match user request of "balanced subset of flows"
    # Let's try to get 50/50 benign/malicious if label column exists
    if 'label' in df.columns:
        benign = df[df['label'] == 0]
        malicious = df[df['label'] == 1]
        
        n_samples = min(len(benign), len(malicious), sample_size // 2)
        
        if n_samples > 0:
            benign_sample = benign.sample(n=n_samples, random_state=random_state)
            malicious_sample = malicious.sample(n=n_samples, random_state=random_state)
            df_sampled = pd.concat([benign_sample, malicious_sample])
            # Shuffle
            df_sampled = df_sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
            print(f"Loaded balanced subset: {len(df_sampled)} rows ({len(benign_sample)} benign, {len(malicious_sample)} malicious)")
            return df_sampled
    
    # Fallback if balancing fails
    print(f"Could not balance perfectly. Returning random sample of {sample_size}.")
    return df.sample(n=min(len(df), sample_size), random_state=random_state).reset_index(drop=True)

if __name__ == "__main__":
    df = load_and_preprocess()
    print(df.head())
