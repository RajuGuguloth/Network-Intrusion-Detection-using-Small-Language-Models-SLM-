import torch
from transformers import RobertaConfig, RobertaForMaskedLM
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

def create_nano_network_model(vocab_size=50000):
    """
    Creates a 'Nano' RoBERTa model configuration.
    This is much smaller than standard BERT/RoBERTa, designed for 
    demonstration and fast training on small datasets.
    """
    config = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=514,
        num_attention_heads=4,
        num_hidden_layers=4,
        type_vocab_size=1,
        hidden_size=256,   # Small vector size (Standard is 768)
        intermediate_size=1024 # Standard is 3072
    )
    
    print("Initializing Nano-RoBERTa Model with config:")
    print(config)
    
    model = RobertaForMaskedLM(config)
    return model

def test_forward_pass():
    print("\n--- Testing Model Forward Pass ---")
    
    # 1. Load Tokenizer
    print("Loading Tokenizer...")
    tokenizer = ByteLevelBPETokenizer(
        "network_tokenizer/vocab.json",
        "network_tokenizer/merges.txt",
    )
    # Add special tokens handling manually for this raw tokenizer, 
    # or just use encode which handles basic BPE. 
    # (HuggingFace 'PreTrainedTokenizerFast' wrappers usually handle this better, 
    # but for raw 'tokenizers' lib we just get IDs).
    
    # 2. Prepare Input
    sample_log = "192.168.1.50 10.0.0.5 TCP 48 54 54 50 20 2F 20 48 54 54 50" # HTTP GET...
    encoded = tokenizer.encode(sample_log)
    
    # Convert to PyTorch Tensor
    input_ids = torch.tensor([encoded.ids])
    attention_mask = torch.tensor([encoded.attention_mask])
    
    print(f"Input Log: {sample_log}")
    print(f"Token IDs: {input_ids}")
    
    # 3. Initialize Model
    model = create_nano_network_model(vocab_size=tokenizer.get_vocab_size())
    
    # 4. Run Inference (Forward Pass)
    output = model(input_ids, attention_mask=attention_mask)
    
    print("\nModel Output:")
    print(f"Logits Shape: {output.logits.shape}")
    print("Success! The model accepted the network tokens and produced a prediction tensor.")
    print("The shape (1, Sequence_Length, Vocab_Size) means it predicted a probability for every token in the vocabulary for each position.")

if __name__ == "__main__":
    test_forward_pass()
