def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."

tokens = text.encode("utf-8")  # raw bytes
tokens = list(map(int, tokens))  # convert to a list of integers
print("---")
print(text)
print("Text length:", len(text))
print("---")
print(tokens)
print("Token (Bytes) length:", len(tokens))

stats = get_stats(tokens)
print("Byte Pair stats:")
print(sorted(((v, k) for k, v in stats.items()), reverse=True))

# Replace all consecutive occurrences of pair with the new token idx
print("Merging pairs:")
print("pair:", (6, 7), "idx:", 99)
print("Before merge:", [5, 6, 6, 7, 9, 1])
print("After merge:", merge([5, 6, 6, 7, 9, 1], [6, 7], 99))

# Replace the most frequent pair with a new token, 256
# (256 is the first unused token id)
top_pair = max(stats, key=stats.get)
tokens2 = merge(tokens, top_pair, 256)
print(tokens2)
print("length:", len(tokens2))

# Merge loop
vocab_size = 276  # desired vocabulary size
num_merges = vocab_size - 256  # number of merges to perform
ids = list(tokens)  # copy tokens to ids

merges = {}  # (int, int) -> int
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")

# Decode
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]  # two bytes object concat


def decode(ids):
    """Decode a list of token ids into a string."""
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")  # avoid UnicodeDecodeError
    return text


print(decode([128]))

print('merges.get:', merges.values())
def encode(text):
    """Encode a string into a list of token ids."""
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


print('encode then decode: hello world! -> ', decode(encode("hello world!")))
print(encode("h"))
print(encode(""))

import regex as re

gpt2pat = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
print(re.findall(gpt2pat, "hello world123 how's are you!!!   "))
print(re.findall(gpt2pat, "hello world123 HOW'S are you!!!   "))

# tiktoken

import tiktoken

# GPT-2
enc = tiktoken.get_encoding("gpt2")
print(enc.encode("    hello world!!!"))

# GPT-4
enc = tiktoken.get_encoding('cl100k_base')
print(enc.encode("    hello world!!!"))

import os, json

with open('encoder.json', 'r') as f:
    encoder = json.load(f)  # equivalent to vocab

with open('vocab.bpe', 'r', encoding='utf-8') as f:
    bpe_data = f.read()

print('bpe_data[0]:', bpe_data.split('\n')[0:2])
print('bpe_data[-1]:', bpe_data.split('\n')[-2:])
bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
# ^--- equivalent to merges

print(len(encoder))
print(len(bpe_merges))
