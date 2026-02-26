#!/usr/bin/env python3
"""Convert NeMo parakeet-tdt_ctc-110m checkpoint to safetensors for axiom.

Usage:
    # Dump NeMo checkpoint keys (discovery):
    python scripts/convert_nemo.py --dump path/to/parakeet-tdt_ctc-110m.nemo

    # Convert to safetensors:
    python scripts/convert_nemo.py path/to/parakeet-tdt_ctc-110m.nemo -o model.safetensors
"""

import argparse
import tarfile
import tempfile
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


NUM_LAYERS = 17
VOCAB_SIZE = 1025
NUM_DURATIONS = 5


# ─── NeMo → Axiom name mapping ──────────────────────────────────────────────

def build_subsampling_map():
    """Map NeMo subsampling conv indices to axiom names.

    NeMo Sequential:
      [0] Conv2d(1, 256, 3, stride=2)     → conv1_
      [1] activation (no params)
      [2] Conv2d(256, 256, 3, groups=256)  → dw1_
      [3] Conv2d(256, 256, 3, stride=2)    → conv2_
      [4] activation (no params)
      [5] Conv2d(256, 256, 3, groups=256)  → dw2_
      [6] Conv2d(256, 256, 3, stride=2)    → conv3_
      [7] activation (no params)
      [8] Conv2d(256, 256, 3, groups=256)  → dw3_
    """
    m = {}
    nemo_to_axiom = {
        "0": "conv1_",
        "2": "dw1_",
        "3": "conv2_",
        "5": "dw2_",
        "6": "conv3_",
        "8": "dw3_",
    }
    for nemo_idx, axiom_name in nemo_to_axiom.items():
        for param in ("weight", "bias"):
            nemo_key = f"encoder.pre_encode.conv.{nemo_idx}.{param}"
            axiom_key = f"encoder_.subsampling_.{axiom_name}.{param}"
            m[nemo_key] = axiom_key

    # Linear projection
    for param in ("weight", "bias"):
        m[f"encoder.pre_encode.out.{param}"] = f"encoder_.subsampling_.proj_.{param}"

    return m


def build_conformer_layer_map(layer_idx):
    """Map NeMo conformer layer keys to axiom keys for layer `layer_idx`."""
    n = f"encoder.layers.{layer_idx}"
    a = f"encoder_.layers_.{layer_idx}"
    m = {}

    # FFN1 (macaron half-step)
    for param in ("weight", "bias"):
        m[f"{n}.norm_feed_forward1.{param}"] = f"{a}.ffn1_.norm_.{param}"
        m[f"{n}.feed_forward1.linear1.{param}"] = f"{a}.ffn1_.fc1_.{param}"
        m[f"{n}.feed_forward1.linear2.{param}"] = f"{a}.ffn1_.fc2_.{param}"

    # Self-attention
    for param in ("weight", "bias"):
        m[f"{n}.norm_self_att.{param}"] = f"{a}.attn_.norm_.{param}"
        m[f"{n}.self_attn.linear_q.{param}"] = f"{a}.attn_.mha_.q_proj.{param}"
        m[f"{n}.self_attn.linear_k.{param}"] = f"{a}.attn_.mha_.k_proj.{param}"
        m[f"{n}.self_attn.linear_v.{param}"] = f"{a}.attn_.mha_.v_proj.{param}"
        m[f"{n}.self_attn.linear_out.{param}"] = f"{a}.attn_.mha_.out_proj.{param}"

    # Positional projection (no bias)
    m[f"{n}.self_attn.linear_pos.weight"] = f"{a}.attn_.pos_proj_.weight"

    # Relative position biases
    m[f"{n}.self_attn.pos_bias_u"] = f"{a}.attn_.pos_bias_u_"
    m[f"{n}.self_attn.pos_bias_v"] = f"{a}.attn_.pos_bias_v_"

    # Conv module — NeMo uses "conv." not "conv_module."
    for param in ("weight", "bias"):
        m[f"{n}.norm_conv.{param}"] = f"{a}.conv_.norm_.{param}"
        m[f"{n}.conv.pointwise_conv1.{param}"] = f"{a}.conv_.pointwise_conv1_.{param}"
        m[f"{n}.conv.depthwise_conv.{param}"] = f"{a}.conv_.depthwise_conv_.{param}"
        m[f"{n}.conv.batch_norm.{param}"] = f"{a}.conv_.batch_norm_.{param}"
        m[f"{n}.conv.pointwise_conv2.{param}"] = f"{a}.conv_.pointwise_conv2_.{param}"

    # BatchNorm running stats
    m[f"{n}.conv.batch_norm.running_mean"] = f"{a}.conv_.batch_norm_.running_mean"
    m[f"{n}.conv.batch_norm.running_var"] = f"{a}.conv_.batch_norm_.running_var"
    m[f"{n}.conv.batch_norm.num_batches_tracked"] = f"{a}.conv_.batch_norm_.num_batches_tracked"

    # FFN2
    for param in ("weight", "bias"):
        m[f"{n}.norm_feed_forward2.{param}"] = f"{a}.ffn2_.norm_.{param}"
        m[f"{n}.feed_forward2.linear1.{param}"] = f"{a}.ffn2_.fc1_.{param}"
        m[f"{n}.feed_forward2.linear2.{param}"] = f"{a}.ffn2_.fc2_.{param}"

    # Final layer norm
    for param in ("weight", "bias"):
        m[f"{n}.norm_out.{param}"] = f"{a}.final_norm_.{param}"

    return m


def build_prediction_map():
    """Map NeMo prediction network keys to axiom keys.

    NeMo path: decoder.prediction.embed / decoder.prediction.dec_rnn.lstm
    """
    m = {}

    # Embedding
    m["decoder.prediction.embed.weight"] = "prediction_.embed_.weight"

    # LSTM (layer 0 only for 110M)
    # NeMo uses nn.LSTM: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
    # Axiom uses LSTMCell: input_proj_.weight, input_proj_.bias, hidden_proj_.weight
    m["decoder.prediction.dec_rnn.lstm.weight_ih_l0"] = "prediction_.lstm_.cells_.0.input_proj_.weight"
    m["decoder.prediction.dec_rnn.lstm.weight_hh_l0"] = "prediction_.lstm_.cells_.0.hidden_proj_.weight"
    # bias_ih and bias_hh are MERGED into input_proj_.bias (handled specially)

    return m


def build_joint_map():
    """Map NeMo TDT joint network keys to axiom keys.

    NeMo structure:
      joint.enc         → encoder projection (with bias)
      joint.pred        → prediction projection (with bias)
      joint.joint_net.2 → combined output [vocab_size + num_durations]
                          Split into label_proj_ and duration_proj_
    """
    m = {}

    for param in ("weight", "bias"):
        m[f"joint.enc.{param}"] = f"tdt_joint_.enc_proj_.{param}"
        m[f"joint.pred.{param}"] = f"tdt_joint_.pred_proj_.{param}"

    # joint.joint_net.2 is combined [1030] = [1025 vocab + 5 durations]
    # Handled specially in convert() — split into label_proj_ and duration_proj_

    return m


def build_ctc_map():
    """Map NeMo CTC decoder keys to axiom keys.

    Try multiple naming patterns since NeMo versions differ.
    """
    m = {}
    # Common patterns for CTC decoder in NeMo
    for prefix in ("ctc_decoder.decoder_layers.0",
                    "ctc_decoder.0"):
        for param in ("weight", "bias"):
            m[f"{prefix}.{param}"] = f"ctc_decoder_.proj_.{param}"
    return m


def build_full_mapping():
    """Build the complete NeMo → axiom mapping."""
    m = {}
    m.update(build_subsampling_map())
    for i in range(NUM_LAYERS):
        m.update(build_conformer_layer_map(i))
    m.update(build_prediction_map())
    m.update(build_joint_map())
    m.update(build_ctc_map())
    return m


# ─── Keys to skip ───────────────────────────────────────────────────────────

SKIP_PREFIXES = (
    "preprocessor.",         # Mel spectrogram filterbank
)

SKIP_KEYS = set()

# pos_bias_u and pos_bias_v are now mapped to ConformerAttention parameters

# LSTM biases are handled specially (merged)
LSTM_BIAS_KEYS = {
    "decoder.prediction.dec_rnn.lstm.bias_ih_l0",
    "decoder.prediction.dec_rnn.lstm.bias_hh_l0",
}

# Combined joint output is handled specially (split)
JOINT_COMBINED_KEYS = {
    "joint.joint_net.2.weight",
    "joint.joint_net.2.bias",
}


def should_skip(key):
    if key in SKIP_KEYS or key in LSTM_BIAS_KEYS or key in JOINT_COMBINED_KEYS:
        return True
    return any(key.startswith(p) for p in SKIP_PREFIXES)


# ─── Extraction ─────────────────────────────────────────────────────────────

def extract_checkpoint(nemo_path):
    """Extract model_weights.ckpt from .nemo tar archive."""
    nemo_path = Path(nemo_path)

    if nemo_path.suffix == ".ckpt":
        return nemo_path

    if nemo_path.suffix != ".nemo":
        # Maybe it's a directory with model_weights.ckpt inside
        ckpt = nemo_path / "model_weights.ckpt"
        if ckpt.exists():
            return ckpt
        raise ValueError(f"Cannot find checkpoint in {nemo_path}")

    # Extract from .nemo tar
    tmpdir = tempfile.mkdtemp()
    with tarfile.open(nemo_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith("model_weights.ckpt"):
                tar.extract(member, tmpdir, filter="data")
                return Path(tmpdir) / member.name

    raise ValueError(f"No model_weights.ckpt found in {nemo_path}")


# ─── Main ───────────────────────────────────────────────────────────────────

def dump_keys(ckpt_path):
    """Print all keys and shapes in the checkpoint."""
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"Total keys: {len(state_dict)}\n")
    for key in sorted(state_dict.keys()):
        t = state_dict[key]
        print(f"  {key:70s} {list(t.shape)}")


def convert(ckpt_path, output_path):
    """Convert NeMo checkpoint to axiom safetensors."""
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mapping = build_full_mapping()

    output = {}
    mapped_nemo_keys = set()
    skipped = []
    unmapped = []

    # ── Special handling: LSTM bias merging ──
    bias_ih = state_dict.get("decoder.prediction.dec_rnn.lstm.bias_ih_l0")
    bias_hh = state_dict.get("decoder.prediction.dec_rnn.lstm.bias_hh_l0")
    if bias_ih is not None and bias_hh is not None:
        merged_bias = bias_ih + bias_hh
        output["prediction_.lstm_.cells_.0.input_proj_.bias"] = merged_bias
        mapped_nemo_keys.update(LSTM_BIAS_KEYS)
        print(f"  Merged LSTM biases: {list(bias_ih.shape)} → {list(merged_bias.shape)}")

    # ── Special handling: split combined joint output ──
    joint_w = state_dict.get("joint.joint_net.2.weight")
    joint_b = state_dict.get("joint.joint_net.2.bias")
    if joint_w is not None:
        # joint_w: [vocab_size + num_durations, joint_hidden]
        # Split: first vocab_size rows → label, last num_durations rows → duration
        output["tdt_joint_.label_proj_.weight"] = joint_w[:VOCAB_SIZE]
        output["tdt_joint_.duration_proj_.weight"] = joint_w[VOCAB_SIZE:]
        mapped_nemo_keys.add("joint.joint_net.2.weight")
        print(f"  Split joint weight: {list(joint_w.shape)} → "
              f"label {list(joint_w[:VOCAB_SIZE].shape)} + "
              f"duration {list(joint_w[VOCAB_SIZE:].shape)}")
    if joint_b is not None:
        output["tdt_joint_.label_proj_.bias"] = joint_b[:VOCAB_SIZE]
        output["tdt_joint_.duration_proj_.bias"] = joint_b[VOCAB_SIZE:]
        mapped_nemo_keys.add("joint.joint_net.2.bias")
        print(f"  Split joint bias: {list(joint_b.shape)} → "
              f"label [{VOCAB_SIZE}] + duration [{NUM_DURATIONS}]")

    # ── Map remaining keys ──
    for nemo_key, tensor in state_dict.items():
        if nemo_key in mapped_nemo_keys:
            continue

        if should_skip(nemo_key):
            skipped.append(nemo_key)
            continue

        if nemo_key in mapping:
            axiom_key = mapping[nemo_key]
            # Avoid duplicates (e.g. CTC with multiple candidate patterns)
            if axiom_key not in output:
                output[axiom_key] = tensor
                mapped_nemo_keys.add(nemo_key)
            else:
                mapped_nemo_keys.add(nemo_key)
        else:
            unmapped.append(nemo_key)

    # Report
    print(f"\nMapped:   {len(mapped_nemo_keys)}")
    print(f"Skipped:  {len(skipped)}")
    print(f"Unmapped: {len(unmapped)}")
    print(f"Output:   {len(output)} tensors")

    if skipped:
        print(f"\nSkipped keys:")
        for k in sorted(skipped):
            print(f"  {k}")

    if unmapped:
        print(f"\nUnmapped keys (ERRORS):")
        for k in sorted(unmapped):
            t = state_dict[k]
            print(f"  {k:70s} {list(t.shape)}")
        print("\nThese NeMo keys have no axiom mapping. Update the converter.")
        sys.exit(1)

    # Check for missing CTC decoder weights
    ctc_missing = []
    for param in ("weight", "bias"):
        key = f"ctc_decoder_.proj_.{param}"
        if key not in output:
            ctc_missing.append(key)
    if ctc_missing:
        print(f"\nNote: CTC decoder weights not found in checkpoint.")
        print(f"  Missing: {ctc_missing}")
        print(f"  CTC head will be randomly initialized at load time.")
        print(f"  (This is normal if the model was trained with TDT only.)")

    # Convert to float32 and save
    output = {k: v.float().contiguous() for k, v in output.items()}
    save_file(output, output_path)

    # Compute total params
    total_params = sum(t.numel() for t in output.values())
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nSaved {output_path} ({file_size_mb:.1f} MB, {total_params:,} parameters)")


def main():
    parser = argparse.ArgumentParser(description="Convert NeMo checkpoint to safetensors")
    parser.add_argument("input", help="Path to .nemo, .ckpt, or directory")
    parser.add_argument("-o", "--output", default="model.safetensors",
                        help="Output safetensors file (default: model.safetensors)")
    parser.add_argument("--dump", action="store_true",
                        help="Just dump checkpoint keys and shapes")
    args = parser.parse_args()

    ckpt_path = extract_checkpoint(args.input)
    print(f"Checkpoint: {ckpt_path}")

    if args.dump:
        dump_keys(ckpt_path)
    else:
        convert(ckpt_path, args.output)


if __name__ == "__main__":
    main()
