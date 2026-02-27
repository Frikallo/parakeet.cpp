#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "safetensors", "packaging"]
# ///
"""Run NeMo inference for reference comparison."""
import torch
import numpy as np

MODEL_PATH = "models/parakeet-tdt_ctc-110m.nemo"
WAV_PATH = "models/2086-149220-0033.wav"

try:
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL_PATH)
    model.eval()

    # Get transcription
    transcriptions = model.transcribe([WAV_PATH])
    print(f"NeMo transcription: {transcriptions}")

    # Also get encoder output stats
    with torch.no_grad():
        # Get preprocessed features
        import wave, struct
        with wave.open(WAV_PATH, 'rb') as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
            samples = struct.unpack(f'<{n}h', raw)

        wav = torch.tensor(samples, dtype=torch.float32).unsqueeze(0) / 32768.0
        wav_len = torch.tensor([wav.shape[1]])

        processed, processed_len = model.preprocessor(input_signal=wav, length=wav_len)
        print(f"\nNeMo features shape: {processed.shape}")
        print(f"  mean={processed.mean().item():.6f}")
        print(f"  min={processed.min().item():.6f}")
        print(f"  max={processed.max().item():.6f}")

        # Get encoder output
        enc_out, enc_len = model.encoder(audio_signal=processed, length=processed_len)
        print(f"\nNeMo encoder shape: {enc_out.shape}")
        print(f"  mean={enc_out.mean().item():.6f}")
        print(f"  min={enc_out.min().item():.6f}")
        print(f"  max={enc_out.max().item():.6f}")

except ImportError:
    print("NeMo not available. Install with: pip install nemo_toolkit[asr]")
    print("Trying lightweight approach with just the checkpoint...")

    # Load checkpoint and run CTC greedy decode manually
    import tarfile, tempfile
    from pathlib import Path
    from safetensors.torch import load_file

    # Just show what the CTC head would produce
    weights = load_file("models/model.safetensors")

    # Print some key weight stats
    for key in sorted(weights.keys()):
        if 'pos_bias' in key:
            t = weights[key]
            print(f"{key}: shape={list(t.shape)} mean={t.mean():.4f} std={t.std():.4f}")
