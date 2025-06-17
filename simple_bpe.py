#!/usr/bin/env python3
"""
Simple Byte Pair Encoding (BPE) implementation

This script demonstrates:
1. Tokenizing text at byte level
2. Building vocabulary by iteratively merging most frequent pairs
3. Encoding and decoding functionality
"""

def get_pair_stats(tokens):
    """Calculate frequency of adjacent token pairs"""
    pair_counts = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts

def merge_tokens(tokens, target_pair, new_token_id):
    """Replace all occurrences of target_pair with new_token_id"""
    result = []
    i = 0
    while i < len(tokens):
        # Check if current and next position match target pair
        if (i < len(tokens) - 1 and 
            tokens[i] == target_pair[0] and 
            tokens[i + 1] == target_pair[1]):
            result.append(new_token_id)
            i += 2  # Skip the pair
        else:
            result.append(tokens[i])
            i += 1
    return result

def train_bpe(text, vocab_size=300):
    """Train BPE model"""
    print(f"=== BPE Training Started ===")
    print(f"Original text: {text}")
    print(f"Target vocabulary size: {vocab_size}")
    
    # Tokenize as UTF-8 bytes
    tokens = list(text.encode('utf-8'))
    print(f"Initial token count: {len(tokens)}")
    
    # Record merge rules
    merges = {}
    
    # Start from 256 basic byte tokens
    next_token_id = 256
    num_merges = vocab_size - 256
    
    print(f"\n=== Performing {num_merges} merges ===")
    
    for merge_step in range(num_merges):
        # Get pair statistics
        pair_stats = get_pair_stats(tokens)
        
        if not pair_stats:
            print("No more pairs to merge")
            break
            
        # Select most frequent pair
        most_frequent_pair = max(pair_stats.items(), key=lambda x: x[1])
        pair, frequency = most_frequent_pair
        
        print(f"Step {merge_step + 1}: Pair {pair} (freq: {frequency}) → Token ID {next_token_id}")
        
        # Perform merge
        tokens = merge_tokens(tokens, pair, next_token_id)
        merges[pair] = next_token_id
        
        next_token_id += 1
    
    print(f"\nTraining completed!")
    print(f"Final token count: {len(tokens)}")
    print(f"Compression ratio: {len(text.encode('utf-8')) / len(tokens):.2f}x")
    
    return merges, tokens

def create_vocabulary(merges):
    """Build vocabulary from merge rules"""
    vocab = {}
    
    # Basic byte tokens (0-255)
    for i in range(256):
        vocab[i] = bytes([i])
    
    # Merged tokens
    for (token1, token2), merged_id in merges.items():
        vocab[merged_id] = vocab[token1] + vocab[token2]
    
    return vocab

def encode_text(text, merges):
    """Encode text using BPE"""
    tokens = list(text.encode('utf-8'))
    
    while len(tokens) >= 2:
        pair_stats = get_pair_stats(tokens)
        
        # Select highest priority pair that exists in learned merge rules
        valid_pairs = [pair for pair in pair_stats if pair in merges]
        
        if not valid_pairs:
            break
            
        # Choose pair that was learned first (lowest merge ID)
        best_pair = min(valid_pairs, key=lambda p: merges[p])
        merge_id = merges[best_pair]
        
        tokens = merge_tokens(tokens, best_pair, merge_id)
    
    return tokens

def decode_tokens(tokens, vocab):
    """Decode token IDs back to text"""
    byte_sequence = b''.join(vocab[token_id] for token_id in tokens)
    return byte_sequence.decode('utf-8', errors='replace')

def main():
    # Sample text (including Japanese and English)
    sample_text = "こんにちは世界！Hello world! 機械学習は面白いですね。Machine learning is fascinating!"
    
    print("=" * 60)
    print("Byte Pair Encoding (BPE) Demo")
    print("=" * 60)
    
    # Train BPE
    merges, trained_tokens = train_bpe(sample_text, vocab_size=280)
    
    # Build vocabulary
    vocab = create_vocabulary(merges)
    
    print(f"\n=== Encoding/Decoding Test ===")
    
    # Test cases
    test_texts = [
        "こんにちは",
        "Hello",
        "機械学習",
        "world"
    ]
    
    for test_text in test_texts:
        # Encode
        encoded = encode_text(test_text, merges)
        # Decode
        decoded = decode_tokens(encoded, vocab)
        
        print(f"Original text: '{test_text}'")
        print(f"Encoded: {encoded}")
        print(f"Decoded: '{decoded}'")
        print(f"Accuracy: {'✓' if test_text == decoded else '✗'}")
        print("-" * 40)
    
    print(f"\nLearned merge rules: {len(merges)}")
    print(f"Final vocabulary size: {len(vocab)}")

if __name__ == "__main__":
    main()