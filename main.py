import sys
import argparse
import pandas as pd
from data_loader import load_and_preprocess, get_data_splits
from slm_baseline.serializer import format_analyst_prompt
from slm_baseline.slm_client import SLMClient
from baseline_model import BaselineModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="SLM Network Intrusion Detection")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of rows to sample (Keep low for dev)")
    parser.add_argument("--shots", type=int, default=0, help="Number of few-shot examples (0 for analyst mode)")
    parser.add_argument("--model", type=str, default="mistral", help="Ollama model name")
    parser.add_argument("--cot", action="store_true", help="Enable Chain-of-Thought reasoning (Implied by Analyst mode)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio")
    return parser.parse_args()

def get_rates(y_true, y_pred):
    if len(y_true) == 0: return 0.0, 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr

def main():
    args = parse_args()
    
    print(f"--- Configurations ---")
    print(f"Model: {args.model}")
    print(f"Sample Size: {args.sample_size}")
    # Analyst mode uses system prompt, not few-shot examples usually
    print(f"Mode: Security Analyst Persona")
    print("-" * 30)

    print("Loading data...")
    df = load_and_preprocess(sample_size=args.sample_size, random_state=42)
    if df is None: return

    train_df, val_df, test_df = get_data_splits(df, 
                                                train_ratio=1.0 - (args.val_ratio + args.test_ratio),
                                                val_ratio=args.val_ratio, 
                                                test_ratio=args.test_ratio)
    
    print(f"Data split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.")
    if len(test_df) == 0:
        print("Error: Test set is empty. Increase sample_size.")
        return

    # --- Baseline Model ---
    print("\n--- Training Baseline Model (Random Forest) ---")
    feature_cols = [c for c in df.columns if c not in ['label', 'attack_cat']]
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_test = test_df[feature_cols]
    y_test = test_df['label']

    rf_model = BaselineModel()
    rf_model.train(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    rf_tpr, rf_fpr = get_rates(y_test, rf_model.predict(X_test))
    
    print("Baseline Random Forest Metrics (Test Set):")
    print(rf_metrics)


    # --- SLM Model ---
    print(f"\n--- Evaluating SLM ({args.model}) ---")
    client = SLMClient(model_name=args.model)
    
    slm_preds = []
    slm_true = []
    total_duration = 0
    
    print("Running inference on test set...")
    
    # Track explanation quality (keyword check for intent)
    intent_keywords = ["scan", "probe", "flood", "brute", "normal", "expected", "standard", "automated", "attack"]
    valid_explanations = 0
    
    for i, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df))):
        # Format: Security Analyst
        prompt = format_analyst_prompt(train_df, row, num_shots=args.shots)
        
        # Debug: Print the first MALICIOUS prompt to verify Weak Supervision
        if row['label'] == 1:
            print("\n[DEBUG] Malicious Sample Prompt (Analyst Mode):")
            print("-" * 50)
            print(prompt)
            print("-" * 50)
            # Only print once
            if 'printed_mal' not in locals():
                 printed_mal = True
            else:
                 # Hack to suppress future prints if we wanted, but i loop continues
                 pass
        
        pred_label_str, explanation, duration = client.classify(prompt)
        total_duration += duration
        
        # Check for intent keywords
        expl_lower = explanation.lower()
        if any(k in expl_lower for k in intent_keywords):
            valid_explanations += 1
        
        # Convert string label to int
        if pred_label_str == "benign":
            pred_label = 0
        elif pred_label_str == "malicious":
            pred_label = 1
        else:
            pred_label = -1 # Error
            
        slm_preds.append(pred_label)
        slm_true.append(row['label'])

    # Compute Metrics
    valid_indices = [i for i, p in enumerate(slm_preds) if p != -1]
    
    if len(valid_indices) < len(slm_preds):
        print(f"Warning: {len(slm_preds) - len(valid_indices)} parsing errors in SLM output.")
        
    y_true_valid = [slm_true[i] for i in valid_indices]
    y_pred_valid = [slm_preds[i] for i in valid_indices]
    
    slm_tpr, slm_fpr = 0.0, 0.0
    
    if len(y_true_valid) > 0:
        slm_acc = accuracy_score(y_true_valid, y_pred_valid)
        slm_prec = precision_score(y_true_valid, y_pred_valid, pos_label=1, zero_division=0)
        slm_rec = recall_score(y_true_valid, y_pred_valid, pos_label=1, zero_division=0)
        slm_f1 = f1_score(y_true_valid, y_pred_valid, pos_label=1, zero_division=0)
        slm_tpr, slm_fpr = get_rates(y_true_valid, y_pred_valid)
    else:
        slm_acc = slm_prec = slm_rec = slm_f1 = 0.0

    avg_time = total_duration / len(test_df) if len(test_df) > 0 else 0

    print("\n--- Final Comparison (Test Set) ---")
    print(f"{'Metric':<20} | {'Random Forest':<15} | {'SLM (' + args.model + ')':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<20} | {rf_metrics['Accuracy']:.4f}          | {slm_acc:.4f}")
    print(f"{'Precision':<20} | {rf_metrics['Precision']:.4f}          | {slm_prec:.4f}")
    print(f"{'Recall (TPR)':<20} | {rf_tpr:.4f}          | {slm_tpr:.4f}")
    print(f"{'F1 Score':<20}  | {rf_metrics['F1']:.4f}          | {slm_f1:.4f}")
    print(f"{'False Pos Rate':<20} | {rf_fpr:.4f}          | {slm_fpr:.4f}")
    print("-" * 60)
    print(f"{'Avg Inference Time':<20} | {'N/A':<15} | {avg_time:.2f}s")
    print(f"{'Explanation Quality':<20} | {'N/A':<15} | {valid_explanations}/{len(test_df)}")

if __name__ == "__main__":
    main()
