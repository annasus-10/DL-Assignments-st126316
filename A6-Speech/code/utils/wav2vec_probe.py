"""Feature extraction (wav2vec2 + raw mel baseline) and linear-probe training on SpeechCommands."""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.model_selection import train_test_split

from utils.mel import compute_log_mel


def load_speechcommands_subset(probe_words, n_per_class=40, root='data/speechcommands'):
    """Build a small balanced subset of SpeechCommands waveforms by label."""
    os.makedirs(root, exist_ok=True)
    sc_dataset = torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True)

    by_label = {w: [] for w in probe_words}
    for i in range(len(sc_dataset)):
        wvf, sr, label, *_ = sc_dataset[i]
        if label in by_label and len(by_label[label]) < n_per_class:
            by_label[label].append(wvf)
        if all(len(v) >= n_per_class for v in by_label.values()):
            break
    return by_label


def extract_wav2vec2_features(by_label, probe_words, device='cpu'):
    """Mean-pooled frozen wav2vec2 features for each clip."""
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

    w2v_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
    w2v_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base', use_safetensors=True).to(device).eval()
    for p in w2v_model.parameters():
        p.requires_grad = False

    feats, labels_list = [], []
    with torch.no_grad():
        for label, clips in by_label.items():
            for wvf in clips:
                inputs = w2v_extractor(wvf.squeeze(0).numpy(), sampling_rate=16000,
                                        return_tensors='pt').to(device)
                out = w2v_model(**inputs).last_hidden_state
                pooled = out.mean(dim=1).squeeze(0).cpu()
                feats.append(pooled)
                labels_list.append(probe_words.index(label))

    X = torch.stack(feats)
    y = torch.tensor(labels_list)
    return X, y


def extract_mel_features(by_label, probe_words):
    """Mean-pooled raw log-mel-spectrogram features for each clip (no pretrained model)."""
    feats, labels_list = [], []
    for label, clips in by_label.items():
        for wvf in clips:
            log_mel = compute_log_mel(wvf, sample_rate=16000)  # (n_mels, T)
            pooled = log_mel.mean(dim=1)  # (n_mels,)
            feats.append(pooled)
            labels_list.append(probe_words.index(label))

    X = torch.stack(feats)
    y = torch.tensor(labels_list)
    return X, y


def train_linear_probe(X, y, n_classes, n_epochs=100, lr=1e-2, test_size=0.3, seed=42):
    """Train a single linear layer on frozen features, return test accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), y.numpy(), test_size=test_size, random_state=seed, stratify=y.numpy())

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    torch.manual_seed(seed)
    linear_probe = nn.Linear(X.shape[1], n_classes)
    opt_probe = torch.optim.Adam(linear_probe.parameters(), lr=lr)

    for epoch in range(n_epochs):
        logits = linear_probe(X_train_t)
        loss = F.cross_entropy(logits, y_train_t)
        opt_probe.zero_grad()
        loss.backward()
        opt_probe.step()

    with torch.no_grad():
        test_acc = (linear_probe(X_test_t).argmax(1) == y_test_t).float().mean().item()
    return test_acc, linear_probe


def plot_tsne(X, y, probe_words, title, save_path=None):
    """t-SNE visualization of pooled features, colored by class."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    proj = TSNE(n_components=2, random_state=42, perplexity=15).fit_transform(X.numpy())
    plt.figure(figsize=(7, 6))
    colors_map = plt.cm.tab10(np.linspace(0, 1, len(probe_words)))
    y_np = y.numpy()
    for i, word in enumerate(probe_words):
        mask = y_np == i
        plt.scatter(proj[mask, 0], proj[mask, 1], c=[colors_map[i]], label=word, alpha=0.7, s=30)
    plt.legend()
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
