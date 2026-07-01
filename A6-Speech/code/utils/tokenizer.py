"""SpeechTokenizer: character-level tokenizer for TTS, with accent-tag special tokens."""
import re


class SpeechTokenizer:
    """
    Character-level tokenizer for TTS.
    Handles: text normalization, special tokens, accent tags.
    """
    ACCENTS = ['[EN-US]', '[EN-BR]', '[EN-INDIA]', '[EN-AU]', '[EN-DEFAULT]']

    def __init__(self):
        # Base vocabulary: printable ASCII
        chars = " !',-.?abcdefghijklmnopqrstuvwxyz"
        self.vocab = {c: i+3 for i, c in enumerate(chars)}
        self.vocab['<PAD>'] = 0
        self.vocab['<BOS>'] = 1
        self.vocab['<EOS>'] = 2
        # Add accent tag tokens
        for i, a in enumerate(self.ACCENTS):
            self.vocab[a] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def normalize(self, text):
        """Basic text normalization: lowercase, expand common abbreviations."""
        text = text.lower()
        text = re.sub(r'dr\.', 'doctor', text)
        text = re.sub(r'mr\.', 'mister', text)
        text = re.sub(r'(\d+)', lambda m: self._num_to_words(int(m.group())), text)
        text = re.sub(r'[^a-z !\',\-.?\[\]]', '', text)
        return text.strip()

    def _num_to_words(self, n):
        words = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
        return words.get(n, str(n))

    def encode(self, text, add_special=True):
        """
        Tokenize text (with optional accent tags).
        Accent tags like [EN-US] are treated as single tokens, not character-by-character.
        """
        tag_pattern = '|'.join(re.escape(a) for a in self.ACCENTS)
        parts = re.split(f'({tag_pattern})', text)
        tokens = []
        if add_special:
            tokens.append(self.vocab['<BOS>'])
        for part in parts:
            if part in self.ACCENTS:
                tokens.append(self.vocab[part])  # accent tag = single token
            else:
                normalized = self.normalize(part)
                for ch in normalized:
                    if ch in self.vocab:
                        tokens.append(self.vocab[ch])
        if add_special:
            tokens.append(self.vocab['<EOS>'])
        return tokens

    def decode(self, ids):
        return ''.join(self.inv_vocab.get(i, '?') for i in ids
                       if i not in (self.vocab['<PAD>'], self.vocab['<BOS>'], self.vocab['<EOS>']))

    def __len__(self):
        return len(self.vocab)