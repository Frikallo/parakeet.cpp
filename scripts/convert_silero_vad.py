#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "safetensors"]
# ///
"""Convert Silero VAD v5 model to safetensors for axiom.

Downloads the model from torch.hub and extracts weights using the known
parameter mapping for the 16kHz configuration.

Usage:
    python scripts/convert_silero_vad.py
    python scripts/convert_silero_vad.py -o silero_vad_v5.safetensors
    python scripts/convert_silero_vad.py --dump  # List model keys
"""

import argparse
import sys

import torch
from safetensors.torch import save_file


def load_silero_vad():
    """Load Silero VAD v5 from torch.hub."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    return model


# Weight mapping: Silero internal name → axiom safetensors name
#
# Silero VAD v5 architecture (16kHz):
#   STFT:    _model.stft.forward_basis_buffer  (258, 1, 256)
#   Encoder: _model.encoder.{0-3}.reparam_conv.{weight,bias}
#   Decoder: _model.decoder.rnn.{weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0}
#            _model.decoder.decoder.2.{weight, bias}  (Conv1d output after ReLU)
WEIGHT_MAP = {
    "_model.stft.forward_basis_buffer": "stft_conv_.weight",
    "_model.encoder.0.reparam_conv.weight": "conv0_.weight",
    "_model.encoder.0.reparam_conv.bias": "conv0_.bias",
    "_model.encoder.1.reparam_conv.weight": "conv1_.weight",
    "_model.encoder.1.reparam_conv.bias": "conv1_.bias",
    "_model.encoder.2.reparam_conv.weight": "conv2_.weight",
    "_model.encoder.2.reparam_conv.bias": "conv2_.bias",
    "_model.encoder.3.reparam_conv.weight": "conv3_.weight",
    "_model.encoder.3.reparam_conv.bias": "conv3_.bias",
    # LSTM: axiom merges bias_ih + bias_hh into single bias
    # v5 uses weight_ih/weight_hh (no _l0 suffix)
    "_model.decoder.rnn.weight_ih": "rnn_.cells_.0.input_proj_.weight",
    "_model.decoder.rnn.weight_hh": "rnn_.cells_.0.hidden_proj_.weight",
    "_model.decoder.rnn.bias_ih": "__lstm_bias_ih",
    "_model.decoder.rnn.bias_hh": "__lstm_bias_hh",
    # Output conv (after ReLU in decoder.decoder sequential)
    "_model.decoder.decoder.2.weight": "out_conv_.weight",
    "_model.decoder.decoder.2.bias": "out_conv_.bias",
}


def convert(model, output_path, dump_only=False):
    """Convert Silero VAD state dict to safetensors."""
    state = {}
    # Get all parameters and buffers
    for name, param in model.named_parameters():
        state[name] = param.detach().cpu()
    for name, buf in model.named_buffers():
        state[name] = buf.detach().cpu()

    if dump_only:
        print(f"{'Key':<60} {'Shape':<20} {'Dtype'}")
        print("-" * 100)
        for key in sorted(state.keys()):
            t = state[key]
            print(f"{key:<60} {str(list(t.shape)):<20} {t.dtype}")
        print(f"\nTotal: {len(state)} tensors")
        return

    # Map weights
    out = {}
    bias_ih = None
    bias_hh = None

    for src_key, dst_key in WEIGHT_MAP.items():
        if src_key not in state:
            print(f"WARNING: Missing key {src_key}", file=sys.stderr)
            continue

        tensor = state[src_key].contiguous().float()

        if dst_key == "__lstm_bias_ih":
            bias_ih = tensor
            continue
        elif dst_key == "__lstm_bias_hh":
            bias_hh = tensor
            continue

        out[dst_key] = tensor

    # Merge LSTM biases (axiom convention: single bias = bias_ih + bias_hh)
    if bias_ih is not None and bias_hh is not None:
        out["rnn_.cells_.0.input_proj_.bias"] = bias_ih + bias_hh
    elif bias_ih is not None:
        out["rnn_.cells_.0.input_proj_.bias"] = bias_ih

    # Verify tensor count
    print(f"Converted {len(out)} tensors:")
    for key in sorted(out.keys()):
        t = out[key]
        print(f"  {key:<40} {str(list(t.shape)):<20}")

    save_file(out, output_path)
    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Silero VAD v5 to safetensors"
    )
    parser.add_argument(
        "-o", "--output",
        default="silero_vad_v5.safetensors",
        help="Output safetensors path (default: silero_vad_v5.safetensors)",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump model keys and exit",
    )
    args = parser.parse_args()

    print("Loading Silero VAD v5 from torch.hub...")
    model = load_silero_vad()
    convert(model, args.output, dump_only=args.dump)


if __name__ == "__main__":
    main()
