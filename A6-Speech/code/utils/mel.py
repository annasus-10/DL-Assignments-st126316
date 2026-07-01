"""Mel spectrogram utilities — the shared bridge representation used across ASR/TTS."""
import os
import urllib.request

import torch
import torchaudio
import torchaudio.transforms as T


def load_waveform(path, target_sr=16000):
    """Load an audio file and resample to target_sr if needed."""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = T.Resample(sr, target_sr)(waveform)
        sr = target_sr
    return waveform, sr


def compute_log_mel(waveform, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80):
    """waveform: (1, N) or (N,) tensor. Returns log-mel spectrogram (n_mels, T)."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel_tf = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                               hop_length=hop_length, n_mels=n_mels)
    mel_spec = mel_tf(waveform).squeeze()
    return torch.log(mel_spec + 1e-9)


def download_sample_wav(out_path='data/sample_speech.wav'):
    """Download the small torchaudio tutorial sample clip used for the Part 2 demo."""
    url = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not os.path.exists(out_path):
        urllib.request.urlretrieve(url, out_path)
    return out_path
