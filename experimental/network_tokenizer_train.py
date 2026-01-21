import os
import random
from tokenizers import ByteLevelBPETokenizer

def generate_synthetic_data(file_path, num_lines=10000):
    """
    Generates a file containing synthetic 'network' logs in the format:
    [SRC_IP] [DST_IP] [PROTO] <HEX_PAYLOAD>
    """
    print(f"Generating {num_lines} lines of synthetic data...")
    
    protocols = ["TCP", "UDP", "ICMP"]
    common_hex_words = ["48 54 54 50", "47 45 54", "50 4F 53 54", "00 00", "FF FF"] # HTTP, GET, POST...
    
    with open(file_path, "w") as f:
        for _ in range(num_lines):
            # Generate IPs
            src_ip = f"192.168.1.{random.randint(1, 255)}"
            dst_ip = f"10.0.0.{random.randint(1, 255)}"
            proto = random.choice(protocols)
            
            # Generate Payload (Mix of random bytes and "common" words to mimic real patterns)
            payload_parts = []
            length = random.randint(5, 50)
            for _ in range(length):
                if random.random() < 0.1:
                    payload_parts.append(random.choice(common_hex_words))
                else:
                    payload_parts.append(f"{random.randint(0, 255):02X}")
            
            payload = " ".join(payload_parts)
            
            # Write line
            # Structural tokens like [SRC_IP] will be added to special tokens later, 
            # but for training data we just put the text.
            line = f"{src_ip} {dst_ip} {proto} {payload}\n"
            f.write(line)
            
    print("Data generation complete.")

def train_tokenizer(data_file):
    print("Training Tokenizer...")
    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=[data_file], vocab_size=50000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
        "<SRC_IP>",
        "<DST_IP>",
        "<PROTO>"
    ])

    # Save files to disk
    if not os.path.exists("network_tokenizer"):
        os.makedirs("network_tokenizer")
    
    tokenizer.save_model("network_tokenizer")
    print("Tokenizer saved to 'network_tokenizer/' directory.")

if __name__ == "__main__":
    data_file = "synthetic_network_logs.txt"
    generate_synthetic_data(data_file)
    train_tokenizer(data_file)
    
    # Test
    from tokenizers.implementations import ByteLevelBPETokenizer
    from tokenizers.processors import BertProcessing

    tokenizer = ByteLevelBPETokenizer(
        "network_tokenizer/vocab.json",
        "network_tokenizer/merges.txt",
    )
    
    # Test encoding
    test_str = "192.168.1.5 10.0.0.9 TCP 48 54 54 50 00 FF"
    encoded = tokenizer.encode(test_str)
    print(f"Test Entry: {test_str}")
    print(f"Encoded IDs: {encoded.ids}")
    print(f"Tokens: {encoded.tokens}")
