"""
Utility Module
Contains utility functions and classes
"""


def levenshtein_distance(s1, s2):
    """Compute edit distance"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


class LabelEncoder:
    """Encode/decode text"""
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        
    def build_vocab(self, texts):
        chars = sorted(list(set("".join(texts))))
        self.char2idx = {char: idx + 1 for idx, char in enumerate(chars)}
        self.idx2char = {idx + 1: char for idx, char in enumerate(chars)}
        self.idx2char[0] = ''
        print(f"✓ Vocab Size: {len(self.char2idx)} characters")
        
    def encode(self, text):
        return [self.char2idx[char] for char in text if char in self.char2idx]
    
    def decode_greedy(self, preds):
        res = []
        prev = None
        for idx in preds:
            if idx != 0 and idx != prev:
                res.append(self.idx2char[idx])
            prev = idx
        return ''.join(res)
    
    def num_classes(self): 
        return len(self.char2idx) + 1
