#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchaudio", "numpy", "safetensors"]
# ///
"""Compare C++ encoder output with PyTorch reference, layer by layer.
Uses safetensors weights directly with PyTorch, no NeMo dependency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
import sys

# Load weights
weights_path = "models/model.safetensors"
f = safe_open(weights_path, framework="pt")
W = {k: f.get_tensor(k) for k in f.keys()}


def get(name):
    return W[name]


# ─── Audio preprocessing (matching C++ audio.cpp) ─────────────────────
def preprocess_audio(wav_path):
    """Load WAV and compute mel features matching C++ implementation."""
    import wave
    import struct

    with wave.open(wav_path, 'rb') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_samples = wf.getnframes()
        raw = wf.readframes(n_samples)
        samples = struct.unpack(f'<{n_samples * n_channels}h', raw)
        waveform = torch.tensor(samples, dtype=torch.float32) / 32768.0
        if n_channels > 1:
            waveform = waveform.view(-1, n_channels).mean(dim=1)

    # Add dither (skip for deterministic comparison)
    # waveform += torch.randn_like(waveform) * 1e-5

    # STFT parameters matching C++ (n_fft=512, hop=160, win=400)
    n_fft = 512
    hop_length = 160
    win_length = 400
    window = torch.hann_window(win_length, periodic=True)

    # Pad window to n_fft
    padded_window = torch.zeros(n_fft)
    padded_window[:win_length] = window

    stft = torch.stft(
        waveform, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=padded_window, return_complex=True, center=True, pad_mode='reflect'
    )
    # stft: (n_fft/2+1, frames)
    magnitudes = stft.abs()
    power = magnitudes ** 2

    # Mel filterbank
    import torchaudio
    n_mels = 80
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=8000.0,
        n_mels=n_mels,
        sample_rate=16000,
        norm="slaney",
        mel_scale="slaney",
    )
    # mel_fb: (n_fft/2+1, n_mels)
    mel_spec = power.T @ mel_fb  # (frames, n_mels)
    mel_spec = mel_spec.T  # (n_mels, frames)

    # Log
    log_mel = torch.log(mel_spec + 1e-6)

    # Per-feature normalization
    mean = log_mel.mean(dim=1, keepdim=True)
    var = log_mel.var(dim=1, keepdim=True, correction=0)
    log_mel = (log_mel - mean) / (var + 1e-5).sqrt()

    # (n_mels, frames) -> (1, frames, n_mels)
    return log_mel.T.unsqueeze(0)


# ─── ConvSubsampling ──────────────────────────────────────────────────
class ConvSubsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256, 3, stride=2, padding=1)
        self.dw1 = nn.Conv2d(256, 256, 3, stride=2, padding=1, groups=256)
        self.conv2 = nn.Conv2d(256, 256, 1, stride=1, padding=0)
        self.dw2 = nn.Conv2d(256, 256, 3, stride=2, padding=1, groups=256)
        self.conv3 = nn.Conv2d(256, 256, 1, stride=1, padding=0)
        self.proj = nn.Linear(256 * 10, 512)  # 80/8=10

        # Load weights
        p = "encoder_.subsampling_."
        self.conv1.weight.data = get(p + "conv1_.weight")
        self.conv1.bias.data = get(p + "conv1_.bias")
        self.dw1.weight.data = get(p + "dw1_.weight")
        self.dw1.bias.data = get(p + "dw1_.bias")
        self.conv2.weight.data = get(p + "conv2_.weight")
        self.conv2.bias.data = get(p + "conv2_.bias")
        self.dw2.weight.data = get(p + "dw2_.weight")
        self.dw2.bias.data = get(p + "dw2_.bias")
        self.conv3.weight.data = get(p + "conv3_.weight")
        self.conv3.bias.data = get(p + "conv3_.bias")
        self.proj.weight.data = get(p + "proj_.weight")
        self.proj.bias.data = get(p + "proj_.bias")

    def forward(self, x):
        # x: (batch, seq, mel_bins)
        x = x.unsqueeze(1)  # (batch, 1, seq, mel_bins)

        x = self.conv1(x)
        x = F.relu(x)

        x = self.dw1(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = self.dw2(x)
        x = self.conv3(x)
        x = F.relu(x)

        # Flatten: (batch, C, T, F) -> (batch, T, C*F)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x = self.proj(x)
        return x


# ─── CTC Decoder ─────────────────────────────────────────────────────
class CTCDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(512, 1025, 1)
        self.proj.weight.data = get("ctc_decoder_.proj_.weight")
        self.proj.bias.data = get("ctc_decoder_.proj_.bias")

    def forward(self, x):
        # x: (batch, seq, hidden)
        x = x.transpose(1, 2)  # (batch, hidden, seq)
        x = self.proj(x)
        x = x.transpose(1, 2)  # (batch, seq, vocab)
        return F.log_softmax(x, dim=-1)


# ─── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    wav_path = "models/2086-149220-0033.wav"

    print("=== Preprocessing ===")
    features = preprocess_audio(wav_path)
    print(f"Features (Python): {features.shape}")
    print(f"  min={features.min().item():.4f} max={features.max().item():.4f} mean={features.mean().item():.6f}")

    # Load C++ features for comparison
    cpp_features = np.fromfile("models/debug_features_cpp.bin", dtype=np.float32)
    cpp_features = cpp_features.reshape(1, 744, 80)
    cpp_features = torch.from_numpy(cpp_features)
    print(f"Features (C++): {cpp_features.shape}")
    print(f"  min={cpp_features.min().item():.4f} max={cpp_features.max().item():.4f} mean={cpp_features.mean().item():.6f}")
    diff = (features - cpp_features).abs()
    print(f"  Max abs diff: {diff.max().item():.6f}")
    print(f"  Mean abs diff: {diff.mean().item():.6f}")

    print("\n=== Subsampling (Python features) ===")
    sub = ConvSubsampling()
    sub.eval()
    with torch.no_grad():
        sub_out_py = sub(features)
    print(f"Subsampling out: {sub_out_py.shape}")
    print(f"  min={sub_out_py.min().item():.4f} max={sub_out_py.max().item():.4f} mean={sub_out_py.mean().item():.6f}")

    print("\n=== Subsampling (C++ features) ===")
    with torch.no_grad():
        sub_out_cpp = sub(cpp_features)
    print(f"Subsampling out: {sub_out_cpp.shape}")
    print(f"  min={sub_out_cpp.min().item():.4f} max={sub_out_cpp.max().item():.4f} mean={sub_out_cpp.mean().item():.6f}")

    # Apply xscale
    d_model = 512
    xscale = d_model ** 0.5
    sub_scaled = sub_out_cpp * xscale
    print(f"After xscale (*{xscale:.2f}): min={sub_scaled.min().item():.4f} max={sub_scaled.max().item():.4f}")

    # CTC decode directly on subsampling output (no conformer layers) for comparison
    print("\n=== CTC on subsampling output (no encoder layers) ===")
    ctc = CTCDecoder()
    ctc.eval()
    with torch.no_grad():
        log_probs = ctc(sub_out_cpp)
        preds = log_probs.argmax(dim=-1)
        log_probs_scaled = ctc(sub_scaled)
        preds_scaled = log_probs_scaled.argmax(dim=-1)

    blank_id = 1024
    print(f"No xscale: {(preds == blank_id).sum().item()} blank / {preds.shape[1]} total")
    print(f"With xscale: {(preds_scaled == blank_id).sum().item()} blank / {preds_scaled.shape[1]} total")

    # Save features for C++ comparison
    np.save("models/debug_features_py.npy", features.numpy())
    np.save("models/debug_sub_out_py.npy", sub_out.numpy())
    print("\nSaved features and subsampling output to models/debug_*.npy")
