# bpe-simple

A simple implementation of Byte Pair Encoding (BPE) tokenization algorithm.

## Usage

### Run the original sample script
```bash
python bpe_sample.py
```

This script demonstrates the complete BPE implementation including:
- Training on Unicode text
- Comparison with GPT-2/GPT-4 tokenizers via tiktoken
- Loading pre-trained GPT-2 vocabulary and merge rules

### Run the simplified BPE demo
```bash
python simple_bpe.py
```

This script provides a cleaner demonstration of:
- Basic BPE training algorithm
- Encoding and decoding functionality
- Japanese and English text processing
- Compression ratio analysis

## Requirements

- Python 3.x
- `regex` library
- `tiktoken` library (for bpe_sample.py)