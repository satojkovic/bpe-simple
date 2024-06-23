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
print("length:", len(text))
print("---")
print(tokens)
print("length:", len(tokens))

stats = get_stats(tokens)
print(sorted(((v, k) for k, v in stats.items()), reverse=True))

# Replace all consecutive occurrences of pair with the new token idx
print(merge([5, 6, 6, 7, 9, 1], [6, 7], 99))
top_pair = max(stats, key=stats.get)
tokens2 = merge(tokens, top_pair, 256)
print(tokens2)
print("length:", len(tokens2))
