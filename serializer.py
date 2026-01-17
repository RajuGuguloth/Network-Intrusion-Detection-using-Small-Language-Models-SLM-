def serialize_row(row):
    """
    Converts a pandas Series (row) into a text description for the SLM.
    Format:
    "Flow: protocol=TCP, service=HTTP, src_bytes=200, dst_bytes=1500, duration=1.3, packets=14, flags=SFP. Label:"
    """
    # Mapping somewhat standard names to readable ones
    # UNSW-NB15 features: proto, service, sbytes, dbytes, dur, spkts, dpkts, state
    
    parts = []
    
    parts.append(f"protocol={row.get('proto', 'unknown')}")
    parts.append(f"service={row.get('service', 'unknown')}")
    parts.append(f"src_bytes={row.get('sbytes', 0)}")
    parts.append(f"dst_bytes={row.get('dbytes', 0)}")
    parts.append(f"duration={float(row.get('dur', 0)):.4f}")
    
    total_packets = int(row.get('spkts', 0)) + int(row.get('dpkts', 0))
    parts.append(f"packets={total_packets}")
    
    parts.append(f"flags={row.get('state', 'unknown')}")
    
    # Construct the instruction
    return f"Flow: {', '.join(parts)}. Label:"

def format_few_shot_prompt(train_rows, target_row, num_shots=3):
    """
    Constructs the full prompt including system instruction and few-shot examples.
    """
    system_instruction = (
        "You are a cybersecurity analyst. Given flow features, decide if the network flow is 'benign' or 'malicious'. "
        "Answer using exactly one word: benign or malicious.\n\n"
    )
    
    examples = ""
    # Select a few balanced examples if possible, or just random `num_shots`
    # We assume train_rows has the 'label' column populated
    
    # Ideally pick some benign and some malicious
    benign_samples = train_rows[train_rows['label'] == 0].sample(n=min(len(train_rows[train_rows['label']==0]), int(num_shots/2)+1))
    malicious_samples = train_rows[train_rows['label'] == 1].sample(n=min(len(train_rows[train_rows['label']==1]), int(num_shots/2)+1))
    
    # Interleave them
    shots = []
    
    # Helper to get label string
    def get_label_str(val):
        return "benign" if val == 0 else "malicious"

    for _, r in benign_samples.iterrows():
        shots.append(f"{serialize_row(r)} {get_label_str(r['label'])}.")
        
    for _, r in malicious_samples.iterrows():
        shots.append(f"{serialize_row(r)} {get_label_str(r['label'])}.")
    
    # Limit to num_shots
    import random
    random.shuffle(shots)
    shots = shots[:num_shots]
    
    examples = "\n".join(shots)
    
    query = serialize_row(target_row)
    
    return f"{system_instruction}{examples}\n{query}"
