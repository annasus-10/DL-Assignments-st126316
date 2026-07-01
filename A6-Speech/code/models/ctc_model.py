"""Toy BiLSTM + CTC model, synthetic frame-to-character task, and training utilities."""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHABET = list('helo wrd')
CHAR2IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 0 reserved for blank
IDX2CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET) + 1
N_MELS = 20

WORDS = ['hello', 'world', 'hero', 'red', 'led', 'doer']


def synthesize_frames(word, frames_per_char=(3, 8)):
    """Each character is stretched across a random number of frames + noise."""
    frames, char_at_frame = [], []
    for ch in word:
        n = random.randint(*frames_per_char)
        base = np.zeros(N_MELS)
        base[CHAR2IDX[ch] % N_MELS] = 3.0
        for _ in range(n):
            frames.append(base + np.random.randn(N_MELS) * 0.5)
            char_at_frame.append(ch)
    return np.stack(frames), char_at_frame


class TinyCTCModel(nn.Module):
    """BiLSTM encoder -> linear -> log-softmax over (blank + alphabet)."""
    def __init__(self, in_dim=N_MELS, hidden=64, vocab=VOCAB_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, vocab)

    def forward(self, x):
        h, _ = self.lstm(x)
        return F.log_softmax(self.fc(h), dim=-1)


def edit_distance(a, b):
    """Levenshtein distance between two strings."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def character_error_rate(pred, target):
    if len(target) == 0:
        return 0.0
    return edit_distance(pred, target) / len(target)


def train_ctc_model(n_steps=300, frames_per_char=(3, 8), lr=1e-2, seed=42):
    """Train the toy CTC model, tracking loss and character error rate per step."""
    from utils.ctc import ctc_collapse, BLANK

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = TinyCTCModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    losses, cers = [], []
    for step in range(n_steps):
        word = random.choice(WORDS)
        frames, _ = synthesize_frames(word, frames_per_char)
        x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
        targets = torch.tensor([CHAR2IDX[c] for c in word], dtype=torch.long)

        log_probs = model(x).transpose(0, 1)
        input_lengths = torch.tensor([x.shape[1]], dtype=torch.long)
        target_lengths = torch.tensor([len(word)], dtype=torch.long)

        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            eval_log_probs = model(x).squeeze(0)
        pred_ids = eval_log_probs.argmax(dim=-1).tolist()
        pred_chars_raw = [IDX2CHAR.get(i, BLANK) if i != 0 else BLANK for i in pred_ids]
        decoded = ctc_collapse(pred_chars_raw)
        cers.append(character_error_rate(decoded, word))
        model.train()

    return model, losses, cers


def greedy_decode_grid(model, words=None, frames_per_char=(3, 8), save_path=None):
    """Visualize raw per-frame greedy predictions and their collapsed decoding for each word."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.ctc import ctc_collapse, BLANK

    if words is None:
        words = WORDS

    model.eval()
    fig, axes = plt.subplots(len(words), 1, figsize=(12, 2 * len(words)))
    if len(words) == 1:
        axes = [axes]

    results = []
    for ax, word in zip(axes, words):
        frames, _ = synthesize_frames(word, frames_per_char)
        x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            log_probs = model(x).squeeze(0)
        pred_ids = log_probs.argmax(dim=-1).tolist()
        pred_chars_raw = [IDX2CHAR.get(i, BLANK) if i != 0 else BLANK for i in pred_ids]
        decoded = ctc_collapse(pred_chars_raw)
        results.append((word, decoded, decoded == word))

        T_ = len(pred_chars_raw)
        colors = plt.cm.tab10(np.linspace(0, 1, len(ALPHABET) + 1))
        for t, ch in enumerate(pred_chars_raw):
            idx = 0 if ch == BLANK else ALPHABET.index(ch) + 1
            ax.bar(t, 1, color=colors[idx], edgecolor='white', linewidth=0.3)
            if ch != BLANK:
                ax.text(t, 0.5, ch, ha='center', va='center', fontsize=8, color='white')
        ax.set_xlim(0, T_); ax.set_ylim(0, 1)
        ax.set_yticks([]); ax.set_xticks([])
        correct = 'correct' if decoded == word else 'wrong'
        ax.set_ylabel(f'"{word}"', rotation=0, labelpad=35, fontsize=10, va='center')
        ax.set_title(f'Raw greedy output ({T_} frames) -> collapsed: "{decoded}"  [{correct}]',
                     fontsize=9, loc='left')

    plt.suptitle('CTC Greedy Decoding: Raw Frame-Level Output -> Collapsed Text', fontsize=13, y=1.0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    model.train()
    return results
