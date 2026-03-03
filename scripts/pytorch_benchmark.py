#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "pyyaml", "nemo_toolkit[asr]"]
# ///
"""Benchmark PyTorch/NeMo inference for comparison with parakeet.cpp."""
import os
import sys
import time
import tarfile
import tempfile
import types
import wave
import struct

# Mock texterrors to work around Python 3.14 binary incompatibility
_mock = types.ModuleType("texterrors")
_mock.align_texts = None
sys.modules["texterrors"] = sys.modules["texterrors_align"] = _mock

import torch
import yaml

NEMO_PATH = "models/parakeet-tdt_ctc-110m.nemo"
WAV_PATH = "models/2086-149220-0033.wav"
LONG_WAV_PATH = "models/test.wav"


def load_wav(path):
    """Load WAV file as float32 tensor."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        raw = wf.readframes(n)
        samples = struct.unpack(f"<{n * ch}h", raw)
    wav = torch.tensor(samples, dtype=torch.float32) / 32768.0
    if ch > 1:
        wav = wav.reshape(-1, ch).mean(dim=1)
    return wav, sr


def fmt_times(times):
    """Format timing list as 'mean +/- std (min-max) ms'."""
    import statistics

    ms = [t * 1000 for t in times]
    if len(ms) == 1:
        return f"{ms[0]:.0f} ms"
    mean = statistics.mean(ms)
    std = statistics.stdev(ms) if len(ms) > 1 else 0
    return f"{mean:.0f} +/- {std:.0f} ms (min={min(ms):.0f}, max={max(ms):.0f})"


def benchmark(nemo_path, wav_path, device="cpu", warmup=2, runs=5):
    """Load NeMo encoder + preprocessor and benchmark."""
    # Extract checkpoint
    with tarfile.open(nemo_path, "r") as tf:
        cfg = yaml.safe_load(tf.extractfile("./model_config.yaml"))
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
            tmp.write(tf.extractfile("./model_weights.ckpt").read())
            ckpt_path = tmp.name

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    os.unlink(ckpt_path)

    from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
    from nemo.collections.asr.modules.audio_preprocessing import (
        AudioToMelSpectrogramPreprocessor,
    )

    # Build preprocessor
    pre_cfg = cfg["preprocessor"]
    preprocessor = AudioToMelSpectrogramPreprocessor(
        **{k: v for k, v in pre_cfg.items() if k != "_target_"}
    )

    # Build encoder
    enc_cfg = cfg["encoder"]
    encoder = ConformerEncoder(
        **{k: v for k, v in enc_cfg.items() if k != "_target_"}
    )

    # Load weights
    pre_state = {
        k.replace("preprocessor.", ""): v
        for k, v in state_dict.items()
        if k.startswith("preprocessor.")
    }
    enc_state = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    preprocessor.load_state_dict(pre_state, strict=False)
    encoder.load_state_dict(enc_state, strict=False)

    preprocessor.eval()
    encoder.eval()

    dev = torch.device(device)
    preprocessor = preprocessor.to(dev)
    encoder = encoder.to(dev)

    # Load audio
    wav, sr = load_wav(wav_path)
    print(f"  Audio: {wav.shape[0]} samples, {sr} Hz, {wav.shape[0]/sr:.2f}s")

    wav_tensor = wav.unsqueeze(0).to(dev)
    wav_len = torch.tensor([wav.shape[0]], device=dev)

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            features, feat_len = preprocessor(
                input_signal=wav_tensor, length=wav_len
            )
            enc_out, enc_len = encoder(audio_signal=features, length=feat_len)
            sync()

        # Benchmark preprocessing
        pre_times = []
        for _ in range(runs):
            sync()
            t0 = time.perf_counter()
            features, feat_len = preprocessor(
                input_signal=wav_tensor, length=wav_len
            )
            sync()
            t1 = time.perf_counter()
            pre_times.append(t1 - t0)

        # Benchmark encoder
        enc_times = []
        for _ in range(runs):
            sync()
            t0 = time.perf_counter()
            enc_out, enc_len = encoder(audio_signal=features, length=feat_len)
            sync()
            t1 = time.perf_counter()
            enc_times.append(t1 - t0)

        # Benchmark end-to-end (preprocess + encode)
        e2e_times = []
        for _ in range(runs):
            sync()
            t0 = time.perf_counter()
            features, feat_len = preprocessor(
                input_signal=wav_tensor, length=wav_len
            )
            enc_out, enc_len = encoder(audio_signal=features, length=feat_len)
            sync()
            t1 = time.perf_counter()
            e2e_times.append(t1 - t0)

    print(f"  Features shape: {features.shape}")
    print(f"  Encoder out: {enc_out.shape}")
    print(f"  Encoder stats: min={enc_out.min():.4f} max={enc_out.max():.4f} mean={enc_out.mean():.4f}")

    return {
        "preprocess": pre_times,
        "encoder": enc_times,
        "e2e": e2e_times,
    }


if __name__ == "__main__":
    wav_files = [WAV_PATH]
    if len(sys.argv) > 1:
        wav_files = sys.argv[1:]
    elif os.path.exists(LONG_WAV_PATH):
        wav_files.append(LONG_WAV_PATH)

    for wav_path in wav_files:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {wav_path}")
        print(f"{'='*60}")

        for device in ["cpu", "mps"]:
            print(f"\n--- PyTorch ({device.upper()}) ---")
            try:
                result = benchmark(
                    NEMO_PATH, wav_path, device=device, warmup=2, runs=5
                )
                print(f"  Preprocess: {fmt_times(result['preprocess'])}")
                print(f"  Encoder:    {fmt_times(result['encoder'])}")
                print(f"  End-to-end: {fmt_times(result['e2e'])}")
            except Exception as e:
                print(f"  Failed: {e}")
