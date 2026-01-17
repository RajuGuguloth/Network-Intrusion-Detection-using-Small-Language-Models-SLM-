import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import load_and_preprocess
from serializer import format_few_shot_prompt
from slm_client import SLMClient
from baseline_model import BaselineModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def main():
    print("Loading data...")
    # Using small sample for demonstration and speed
    df = load_and_preprocess(sample_size=20, random_state=42)
    if df is None:
        print("Failed to load data.")
        return

    # Split Data
    # We need a hold-out set for testing, and a training set to pull few-shot examples from
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)
    print(f"Data split: {len(train_df)} train (for few-shot), {len(test_df)} test.")

    # --- Baseline Model ---
    print("\n--- Training Baseline Model (Random Forest) ---")
    # Prepare features for RF (drop serialized text if any, keep numericals/categoricals)
    # The data_loader selects: 'proto', 'service', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'dur', 'state', 'attack_cat', 'label'
    
    feature_cols = [c for c in df.columns if c not in ['label', 'attack_cat']]
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_test = test_df[feature_cols]
    y_test = test_df['label']

    rf_model = BaselineModel()
    rf_model.train(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    print("Baseline Random Forest Metrics:")
    print(rf_metrics)


    # --- SLM Model ---
    print("\n--- Evaluating SLM (Mistral/Ollama) ---")
    client = SLMClient()
    
    slm_preds = []
    slm_true = []
    
    print("Running inference on test set...")
    # Identify label mapping
    # 0 -> benign, 1 -> malicious
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # Construct Prompt
        prompt = format_few_shot_prompt(train_df, row, num_shots=3)
        
        # Call SLM
        prediction_text = client.classify(prompt)
        
        # Map text back to 0/1
        # 'benign' -> 0, 'malicious' -> 1
        # If unknown, treat as incorrect (or maybe default to benign? let's say -1 for error)
        if "benign" in prediction_text:
            pred_label = 0
        elif "malicious" in prediction_text:
            pred_label = 1
        else:
            pred_label = -1 # incorrect/parse error
            
        slm_preds.append(pred_label)
        slm_true.append(row['label'])

    # Compute Metrics for SLM
    # Filter out -1 if we want strict comparison, or treat as error
    valid_indices = [i for i, p in enumerate(slm_preds) if p != -1]
    
    if len(valid_indices) < len(slm_preds):
        print(f"Warning: {len(slm_preds) - len(valid_indices)} parsing errors in SLM output.")
        
    y_true_valid = [slm_true[i] for i in valid_indices]
    y_pred_valid = [slm_preds[i] for i in valid_indices]
    
    if len(y_true_valid) > 0:
        slm_acc = accuracy_score(y_true_valid, y_pred_valid)
        slm_prec = precision_score(y_true_valid, y_pred_valid, pos_label=1, zero_division=0)
        slm_rec = recall_score(y_true_valid, y_pred_valid, pos_label=1, zero_division=0)
        slm_f1 = f1_score(y_true_valid, y_pred_valid, pos_label=1, zero_division=0)
    else:
        slm_acc = slm_prec = slm_rec = slm_f1 = 0.0

    print("\n--- Final Comparison ---")
    print(f"{'Metric':<15} | {'Random Forest':<15} | {'SLM (Few-Shot)':<15}")
    print("-" * 50)
    print(f"{'Accuracy':<15} | {rf_metrics['Accuracy']:.4f}          | {slm_acc:.4f}")
    print(f"{'Precision':<15} | {rf_metrics['Precision']:.4f}          | {slm_prec:.4f}")
    print(f"{'Recall':<15}    | {rf_metrics['Recall']:.4f}          | {slm_rec:.4f}")
    print(f"{'F1 Score':<15}  | {rf_metrics['F1']:.4f}          | {slm_f1:.4f}")

if __name__ == "__main__":
    main()
