import numpy as np
import torch
from rag_audio.indexer import compute_features


def test_rms_known_signal():
    signal = torch.ones(16000)  # flat amplitude
    features = compute_features(signal, sr=16000)
    assert 0.99 < features['rms'] < 1.01


def test_dominant_freq_sine_wave():
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    signal = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 1.0, sr))
    features = compute_features(signal, sr)
    assert 430 < features['dominant_freq_hz'] < 450
