# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a simple implementation of Byte Pair Encoding (BPE) tokenization, demonstrating the core concepts behind subword tokenization used in language models like GPT-2 and GPT-4.

## Architecture

The codebase consists of a single Python script (`bpe_sample.py`) that implements:

1. **Core BPE Functions**:
   - `get_stats(ids)`: Counts frequency of adjacent token pairs
   - `merge(ids, pair, idx)`: Merges all occurrences of a token pair with a new token ID
   - `encode(text)`: Converts text to token IDs using learned merges
   - `decode(ids)`: Converts token IDs back to text

2. **Training Process**: 
   - Starts with UTF-8 byte-level tokens (0-255)
   - Iteratively merges most frequent pairs to build vocabulary up to desired size
   - Stores merge rules in `merges` dictionary

3. **External Data Files**:
   - `encoder.json`: Pre-trained vocabulary mapping (token -> ID)
   - `vocab.bpe`: Pre-trained merge rules from GPT-2

## Dependencies

The script requires:
- `regex` library for pattern matching
- `tiktoken` library for GPT tokenizer comparisons

## Running the Code

Execute the main script:
```bash
python bpe_sample.py
```

The script demonstrates:
- BPE training on Unicode text
- Encoding/decoding examples
- Comparison with GPT-2/GPT-4 tokenizers via tiktoken
- Loading pre-trained GPT-2 vocabulary and merge rules

## Key Implementation Details

- Uses byte-level encoding as the base (256 initial tokens)
- Implements greedy merge strategy during encoding
- Handles Unicode text through UTF-8 encoding
- Includes regex pattern for GPT-2 style text splitting