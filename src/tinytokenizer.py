""" Build a trainable BPE-style tokenizer from scratch """

class TinyEncoder:
    def __init__(self, merges):
        self.merges = merges
    
    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            stats = TinyTokenizer.get_stats(tokens)
            # pick the earliest merge (lowest rank)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = TinyTokenizer.merge(tokens, pair, idx)
        return tokens


class TinyDecoder:
    def __init__(self, vocab):
        self.vocab = vocab
    
    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")


class TinyTokenizer:
    def __init__(self, vocab_size=276):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.encoder = None
        self.decoder = None
    
    def train(self, text):
        """Train tokenizer merges from a text corpus"""
        ids = list(text.encode("utf-8"))
        merges = {}
        num_merges = self.vocab_size - 256
        
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)  # most frequent pair
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx
            # debug print
            print(f"[{i+1}] merging {pair} -> {idx}")
        
        self.merges = merges
        self.vocab = self._build_vocab()
        self.encoder = TinyEncoder(self.merges)
        self.decoder = TinyDecoder(self.vocab)
    
    def encode(self, text):
        return self.encoder.encode(text)
    
    def decode(self, ids):
        return self.decoder.decode(ids)

    # ----------- static helpers ------------
    @classmethod
    def get_stats(cls, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    @classmethod
    def merge(cls, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab
