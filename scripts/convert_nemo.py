#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "safetensors", "packaging", "numpy"]
# ///
"""Convert NeMo Parakeet checkpoints to safetensors for axiom.

Supported models:
  - 110m-tdt-ctc  (parakeet-tdt_ctc-110m)
  - 600m-tdt      (parakeet-tdt-0.6b-v2, multilingual)
  - eou-120m      (parakeet-tdt_ctc-0.12b-eou)
  - nemotron-600m (parakeet-tdt-1.1b / nemotron streaming)

Usage:
    # Dump NeMo checkpoint keys (discovery):
    python scripts/convert_nemo.py --dump path/to/model.nemo

    # Convert to safetensors:
    python scripts/convert_nemo.py path/to/model.nemo -o model.safetensors

    # Convert specific model type:
    python scripts/convert_nemo.py path/to/model.nemo -o model.safetensors --model 600m-tdt
"""

import argparse
import tarfile
import tempfile
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


# ─── Model Presets ───────────────────────────────────────────────────────────

MODEL_PRESETS = {
    "110m-tdt-ctc": {
        "num_layers": 17,
        "vocab_size": 1025,
        "num_durations": 5,
        "num_lstm_layers": 1,
        "has_ctc": True,
        "joint_prefix": "tdt_joint_",
    },
    "600m-tdt": {
        "num_layers": 24,
        "vocab_size": 8193,
        "num_durations": 5,
        "num_lstm_layers": 2,
        "has_ctc": False,
        "joint_prefix": "joint_",
    },
    "rnnt-600m": {
        "num_layers": 24,
        "vocab_size": 1025,
        "num_durations": 0,
        "num_lstm_layers": 2,
        "has_ctc": False,
        "joint_prefix": "joint_",
        "is_rnnt": True,  # joint output = vocab only (no duration split)
    },
    "eou-120m": {
        "num_layers": 17,
        "vocab_size": 1025,
        "num_durations": 5,
        "num_lstm_layers": 1,
        "has_ctc": True,
        "joint_prefix": "tdt_joint_",
    },
    "nemotron-600m": {
        "num_layers": 24,
        "vocab_size": 8193,
        "num_durations": 5,
        "num_lstm_layers": 2,
        "has_ctc": False,
        "joint_prefix": "joint_",
    },
    "sortformer": {
        "num_layers": 17,
        "num_transformer_layers": 18,
        "vocab_size": 0,
        "num_durations": 0,
        "num_lstm_layers": 0,
        "has_ctc": False,
        "has_decoder": False,
        "joint_prefix": "",
        "encoder_prefix": "nest_encoder_",
    },
}

DEFAULT_MODEL = "110m-tdt-ctc"

# For backwards compatibility
NUM_LAYERS = 17
VOCAB_SIZE = 1025
NUM_DURATIONS = 5


# ─── NeMo → Axiom name mapping ──────────────────────────────────────────────

def build_subsampling_map(axiom_prefix="encoder_"):
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
            axiom_key = f"{axiom_prefix}.subsampling_.{axiom_name}.{param}"
            m[nemo_key] = axiom_key

    # Linear projection
    for param in ("weight", "bias"):
        m[f"encoder.pre_encode.out.{param}"] = f"{axiom_prefix}.subsampling_.proj_.{param}"

    return m


def build_conformer_layer_map(layer_idx, axiom_prefix="encoder_"):
    """Map NeMo conformer layer keys to axiom keys for layer `layer_idx`."""
    n = f"encoder.layers.{layer_idx}"
    a = f"{axiom_prefix}.layers_.{layer_idx}"
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


def build_prediction_map(num_lstm_layers=1):
    """Map NeMo prediction network keys to axiom keys.

    NeMo path: decoder.prediction.embed / decoder.prediction.dec_rnn.lstm
    """
    m = {}

    # Embedding
    m["decoder.prediction.embed.weight"] = "prediction_.embed_.weight"

    # LSTM layers
    for l in range(num_lstm_layers):
        m[f"decoder.prediction.dec_rnn.lstm.weight_ih_l{l}"] = f"prediction_.lstm_.cells_.{l}.input_proj_.weight"
        m[f"decoder.prediction.dec_rnn.lstm.weight_hh_l{l}"] = f"prediction_.lstm_.cells_.{l}.hidden_proj_.weight"
        # bias_ih and bias_hh are MERGED into input_proj_.bias (handled specially)

    return m


def build_joint_map(joint_prefix="tdt_joint_"):
    """Map NeMo TDT joint network keys to axiom keys.

    NeMo structure:
      joint.enc         → encoder projection (with bias)
      joint.pred        → prediction projection (with bias)
      joint.joint_net.2 → combined output [vocab_size + num_durations]
                          Split into label_proj_ and duration_proj_
    """
    m = {}

    for param in ("weight", "bias"):
        m[f"joint.enc.{param}"] = f"{joint_prefix}.enc_proj_.{param}"
        m[f"joint.pred.{param}"] = f"{joint_prefix}.pred_proj_.{param}"

    # joint.joint_net.2 is combined [vocab_size + num_durations]
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


def build_transformer_encoder_map(num_layers):
    """Map NeMo transformer encoder layers to axiom TransformerEncoder keys.

    NeMo structure per layer:
      transformer_encoder.layers.N.first_sub_layer  = MHA (query_net, key_net, value_net, out_projection)
      transformer_encoder.layers.N.layer_norm_1     = pre-MHA norm
      transformer_encoder.layers.N.layer_norm_2     = pre-FFN norm
      transformer_encoder.layers.N.second_sub_layer = FFN (dense_in, dense_out)
    """
    m = {}
    for i in range(num_layers):
        n = f"transformer_encoder.layers.{i}"
        a = f"transformer_.layers_.{i}"

        for param in ("weight", "bias"):
            # Layer norms
            m[f"{n}.layer_norm_1.{param}"] = f"{a}.norm1_.{param}"
            m[f"{n}.layer_norm_2.{param}"] = f"{a}.norm2_.{param}"

            # MHA projections
            m[f"{n}.first_sub_layer.query_net.{param}"] = f"{a}.mha_.q_proj.{param}"
            m[f"{n}.first_sub_layer.key_net.{param}"] = f"{a}.mha_.k_proj.{param}"
            m[f"{n}.first_sub_layer.value_net.{param}"] = f"{a}.mha_.v_proj.{param}"
            m[f"{n}.first_sub_layer.out_projection.{param}"] = f"{a}.mha_.out_proj.{param}"

            # FFN
            m[f"{n}.second_sub_layer.dense_in.{param}"] = f"{a}.fc1_.{param}"
            m[f"{n}.second_sub_layer.dense_out.{param}"] = f"{a}.fc2_.{param}"

    return m


def build_sortformer_modules_map():
    """Map NeMo sortformer_modules to axiom Sortformer keys."""
    m = {}
    for param in ("weight", "bias"):
        m[f"sortformer_modules.encoder_proj.{param}"] = f"projection_.{param}"
        m[f"sortformer_modules.single_hidden_to_spks.{param}"] = f"output_proj_.{param}"
        m[f"sortformer_modules.first_hidden_to_hidden.{param}"] = f"first_hidden_.{param}"
        m[f"sortformer_modules.hidden_to_spks.{param}"] = f"hidden_to_spks_.{param}"
    return m


def build_full_mapping(preset=None):
    """Build the complete NeMo → axiom mapping."""
    if preset is None:
        preset = MODEL_PRESETS[DEFAULT_MODEL]

    m = {}
    encoder_prefix = preset.get("encoder_prefix", "encoder_")
    m.update(build_subsampling_map(encoder_prefix))
    for i in range(preset["num_layers"]):
        m.update(build_conformer_layer_map(i, encoder_prefix))

    # Models without decoder (e.g., sortformer) skip prediction/joint mapping
    has_decoder = preset.get("has_decoder", True)
    if has_decoder and preset["num_lstm_layers"] > 0:
        m.update(build_prediction_map(preset["num_lstm_layers"]))
    if has_decoder and preset["joint_prefix"]:
        m.update(build_joint_map(preset["joint_prefix"]))
    if preset.get("has_ctc", False):
        m.update(build_ctc_map())

    # Sortformer-specific mappings
    num_transformer_layers = preset.get("num_transformer_layers", 0)
    if num_transformer_layers > 0:
        m.update(build_transformer_encoder_map(num_transformer_layers))
        m.update(build_sortformer_modules_map())

    return m


# ─── Keys to skip ───────────────────────────────────────────────────────────

SKIP_PREFIXES = (
    "preprocessor.",         # Mel spectrogram filterbank
)

SKIP_KEYS = set()

# pos_bias_u and pos_bias_v are now mapped to ConformerAttention parameters

def get_lstm_bias_keys(num_lstm_layers=1):
    """Get LSTM bias keys for all layers (handled specially: merged)."""
    keys = set()
    for l in range(num_lstm_layers):
        keys.add(f"decoder.prediction.dec_rnn.lstm.bias_ih_l{l}")
        keys.add(f"decoder.prediction.dec_rnn.lstm.bias_hh_l{l}")
    return keys

# For backwards compatibility
LSTM_BIAS_KEYS = get_lstm_bias_keys(1)

# Combined joint output is handled specially (split)
JOINT_COMBINED_KEYS = {
    "joint.joint_net.2.weight",
    "joint.joint_net.2.bias",
}


def should_skip(key, lstm_bias_keys=None):
    if lstm_bias_keys is None:
        lstm_bias_keys = LSTM_BIAS_KEYS
    if key in SKIP_KEYS or key in lstm_bias_keys or key in JOINT_COMBINED_KEYS:
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


def convert(ckpt_path, output_path, model_type=DEFAULT_MODEL):
    """Convert NeMo checkpoint to axiom safetensors."""
    preset = MODEL_PRESETS[model_type]
    vocab_size = preset["vocab_size"]
    num_durations = preset["num_durations"]
    num_lstm_layers = preset["num_lstm_layers"]
    joint_prefix = preset["joint_prefix"]
    lstm_bias_keys = get_lstm_bias_keys(num_lstm_layers)

    print(f"Model type: {model_type}")
    print(f"  Encoder layers: {preset['num_layers']}, vocab: {vocab_size}, "
          f"LSTM layers: {num_lstm_layers}, CTC: {preset.get('has_ctc', False)}")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mapping = build_full_mapping(preset)

    output = {}
    mapped_nemo_keys = set()
    skipped = []
    unmapped = []

    # ── Special handling: LSTM bias merging (all layers) ──
    for l in range(num_lstm_layers):
        bias_ih = state_dict.get(f"decoder.prediction.dec_rnn.lstm.bias_ih_l{l}")
        bias_hh = state_dict.get(f"decoder.prediction.dec_rnn.lstm.bias_hh_l{l}")
        if bias_ih is not None and bias_hh is not None:
            merged_bias = bias_ih + bias_hh
            output[f"prediction_.lstm_.cells_.{l}.input_proj_.bias"] = merged_bias
            mapped_nemo_keys.add(f"decoder.prediction.dec_rnn.lstm.bias_ih_l{l}")
            mapped_nemo_keys.add(f"decoder.prediction.dec_rnn.lstm.bias_hh_l{l}")
            print(f"  Merged LSTM layer {l} biases: {list(bias_ih.shape)} → {list(merged_bias.shape)}")

    # ── Special handling: joint output ──
    is_rnnt = preset.get("is_rnnt", False)
    joint_w = state_dict.get("joint.joint_net.2.weight")
    joint_b = state_dict.get("joint.joint_net.2.bias")
    if joint_w is not None:
        if is_rnnt:
            # RNNT: joint output is just vocab (no duration split)
            output[f"{joint_prefix}.out_proj_.weight"] = joint_w
            mapped_nemo_keys.add("joint.joint_net.2.weight")
            print(f"  RNNT joint weight: {list(joint_w.shape)} → out_proj_")
        else:
            output[f"{joint_prefix}.label_proj_.weight"] = joint_w[:vocab_size]
            output[f"{joint_prefix}.duration_proj_.weight"] = joint_w[vocab_size:]
            mapped_nemo_keys.add("joint.joint_net.2.weight")
            print(f"  Split joint weight: {list(joint_w.shape)} → "
                  f"label {list(joint_w[:vocab_size].shape)} + "
                  f"duration {list(joint_w[vocab_size:].shape)}")
    if joint_b is not None:
        if is_rnnt:
            output[f"{joint_prefix}.out_proj_.bias"] = joint_b
            mapped_nemo_keys.add("joint.joint_net.2.bias")
            print(f"  RNNT joint bias: {list(joint_b.shape)} → out_proj_")
        else:
            output[f"{joint_prefix}.label_proj_.bias"] = joint_b[:vocab_size]
            output[f"{joint_prefix}.duration_proj_.bias"] = joint_b[vocab_size:]
            mapped_nemo_keys.add("joint.joint_net.2.bias")
            print(f"  Split joint bias: {list(joint_b.shape)} → "
                  f"label [{vocab_size}] + duration [{num_durations}]")

    # ── Map remaining keys ──
    for nemo_key, tensor in state_dict.items():
        if nemo_key in mapped_nemo_keys:
            continue

        if should_skip(nemo_key, lstm_bias_keys):
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

    # Check for missing CTC decoder weights (only if model has CTC)
    if preset.get("has_ctc", False):
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
    parser.add_argument("--model", choices=list(MODEL_PRESETS.keys()),
                        default=DEFAULT_MODEL,
                        help=f"Model type (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    ckpt_path = extract_checkpoint(args.input)
    print(f"Checkpoint: {ckpt_path}")

    if args.dump:
        dump_keys(ckpt_path)
    else:
        convert(ckpt_path, args.output, args.model)


if __name__ == "__main__":
    main()
