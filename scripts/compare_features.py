"""Compare C++ and Python mel features with NeMo-compatible preprocessing."""
import torch
import torchaudio
import numpy as np
import wave
import struct

WAV_PATH = "models/2086-149220-0033.wav"

# Load audio
with wave.open(WAV_PATH, 'rb') as wf:
    sr = wf.getframerate()
    n_ch = wf.getnchannels()
    n = wf.getnframes()
    raw = wf.readframes(n)
    samples = struct.unpack(f'<{n * n_ch}h', raw)
    waveform = torch.tensor(samples, dtype=torch.float32) / 32768.0
    if n_ch > 1:
        waveform = waveform.view(-1, n_ch).mean(dim=1)
print(f"Audio: sr={sr}, samples={waveform.shape[0]}")

# NeMo-compatible preprocessing
n_fft = 512
win_length = 400
hop_length = 160
n_mels = 80

# 1. Preemphasis
x = torch.cat([waveform[:1], waveform[1:] - 0.97 * waveform[:-1]])

# 2. STFT (symmetric hann, win_length=400 padded to n_fft=512)
window = torch.hann_window(win_length, periodic=False)
padded_window = torch.zeros(n_fft)
padded_window[:win_length] = window

stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                   window=padded_window, return_complex=True,
                   center=True, pad_mode='reflect')
power = stft.abs() ** 2
print(f"Power: {power.shape}")

# 3. Mel filterbank (slaney)
mel_fb = torchaudio.functional.melscale_fbanks(
    n_freqs=n_fft // 2 + 1, f_min=0.0, f_max=float(sr) / 2.0,
    n_mels=n_mels, sample_rate=sr, norm="slaney", mel_scale="slaney",
)
mel_spec = mel_fb.T @ power  # (n_mels, n_frames)

# 4. Log
log_mel = torch.log(mel_spec + 2.0 ** -24)

# 5. Per-feature normalize (unbiased std, N-1)
mean = log_mel.mean(dim=1, keepdim=True)
std = log_mel.std(dim=1, keepdim=True, correction=1)
norm = (log_mel - mean) / (std + 1e-5)

features_py = norm.T.unsqueeze(0)  # (1, n_frames, n_mels)
print(f"\nPython features: {features_py.shape}")
print(f"  min={features_py.min().item():.4f} max={features_py.max().item():.4f} mean={features_py.mean().item():.6f}")

# C++ features
cpp = np.fromfile("models/debug_features_cpp.bin", dtype=np.float32).reshape(1, 744, 80)
features_cpp = torch.from_numpy(cpp)
print(f"\nC++ features: {features_cpp.shape}")
print(f"  min={features_cpp.min().item():.4f} max={features_cpp.max().item():.4f} mean={features_cpp.mean().item():.6f}")

# Diff
diff = (features_py - features_cpp).abs()
print(f"\nDifference:")
print(f"  Max abs diff: {diff.max().item():.6f}")
print(f"  Mean abs diff: {diff.mean().item():.6f}")

# Sample values
print(f"\nSample values (frame 0, bins 0-4):")
print(f"  Python: {features_py[0, 0, :5].tolist()}")
print(f"  C++:    {features_cpp[0, 0, :5].tolist()}")
print(f"  Python: {features_py[0, 0, 75:80].tolist()}")
print(f"  C++:    {features_cpp[0, 0, 75:80].tolist()}")
