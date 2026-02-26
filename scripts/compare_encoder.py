"""Compare C++ encoder intermediate outputs with PyTorch reference.
Uses same weights from safetensors to build PyTorch reference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open

# Load weights
weights_path = "models/model.safetensors"
f = safe_open(weights_path, framework="pt")
W = {k: f.get_tensor(k) for k in f.keys()}

def get(name):
    return W[name]

# Load Python features (same as fed to C++)
features = np.load("models/debug_features_py.npy")
features = torch.from_numpy(features)  # (1, 744, 80)
print(f"Features: {features.shape}")

# ─── ConvSubsampling (matching C++ architecture) ─────────────────────
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
        x = F.silu(x)
        self.after_conv1 = x.clone()

        x = self.dw1(x)
        x = self.conv2(x)
        x = F.silu(x)
        self.after_block1 = x.clone()

        x = self.dw2(x)
        x = self.conv3(x)
        x = F.silu(x)
        self.after_block2 = x.clone()

        # Flatten: (batch, C, T, F) -> (batch, T, C*F)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x = self.proj(x)
        return x


def compare(name, cpp_path, py_tensor):
    """Compare C++ binary dump with PyTorch tensor."""
    cpp_data = np.fromfile(cpp_path, dtype=np.float32)
    py_np = py_tensor.detach().numpy().flatten()

    if cpp_data.shape != py_np.shape:
        print(f"  {name}: SHAPE MISMATCH cpp={cpp_data.shape} py={py_np.shape}")
        return

    diff = np.abs(cpp_data - py_np)
    print(f"  {name}:")
    print(f"    C++ range: [{cpp_data.min():.6f}, {cpp_data.max():.6f}] mean={cpp_data.mean():.6f}")
    print(f"    Py  range: [{py_np.min():.6f}, {py_np.max():.6f}] mean={py_np.mean():.6f}")
    print(f"    Max abs diff: {diff.max():.6f}")
    print(f"    Mean abs diff: {diff.mean():.6f}")

    # Show where biggest differences are
    if diff.max() > 0.01:
        worst_idx = np.argmax(diff)
        print(f"    Worst at idx {worst_idx}: cpp={cpp_data[worst_idx]:.6f} py={py_np[worst_idx]:.6f}")

        # Show first 10 values
        print(f"    First 10 cpp: {cpp_data[:10]}")
        print(f"    First 10 py:  {py_np[:10]}")


# Run PyTorch subsampling
sub = ConvSubsampling()
sub.eval()
with torch.no_grad():
    sub_out = sub(features)

print(f"\nSubsampling output: {sub_out.shape}")

# Compare at each stage
print("\n=== Stage comparisons ===")
compare("after_conv1", "models/debug_after_conv1.bin", sub.after_conv1)
compare("after_block1", "models/debug_after_block1.bin", sub.after_block1)
compare("after_block2", "models/debug_after_block2.bin", sub.after_block2)
compare("subsampling_out", "models/debug_subsampling_out.bin", sub_out)

# ─── Single ConformerBlock Reference ─────────────────────────────────
print("\n=== Conformer Layer 0 ===")

def sinusoidal_position_embedding(seq_len, d_model):
    """Match C++ sinusoidal_position_embedding."""
    total = 2 * seq_len - 1
    pe = torch.zeros(total, d_model)
    for pos_idx in range(total):
        position = float(seq_len - 1 - pos_idx)
        for i in range(0, d_model, 2):
            div_term = np.exp(float(i) * (-np.log(10000.0) / d_model))
            pe[pos_idx, i] = np.sin(position * div_term)
            if i + 1 < d_model:
                pe[pos_idx, i + 1] = np.cos(position * div_term)
    return pe


class FeedForward(nn.Module):
    def __init__(self, prefix):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)

        self.norm.weight.data = get(prefix + "norm_.weight")
        self.norm.bias.data = get(prefix + "norm_.bias")
        self.fc1.weight.data = get(prefix + "fc1_.weight")
        self.fc1.bias.data = get(prefix + "fc1_.bias")
        self.fc2.weight.data = get(prefix + "fc2_.weight")
        self.fc2.bias.data = get(prefix + "fc2_.bias")

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = F.silu(h)
        h = self.fc2(h)
        return x + h * 0.5


class ConformerConvModule(nn.Module):
    def __init__(self, prefix):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.pw1 = nn.Conv1d(512, 1024, 1)
        self.dw = nn.Conv1d(512, 512, 9, padding=4, groups=512)
        self.bn = nn.BatchNorm1d(512)
        self.pw2 = nn.Conv1d(512, 512, 1)

        self.norm.weight.data = get(prefix + "norm_.weight")
        self.norm.bias.data = get(prefix + "norm_.bias")
        self.pw1.weight.data = get(prefix + "pointwise_conv1_.weight")
        self.pw1.bias.data = get(prefix + "pointwise_conv1_.bias")
        self.dw.weight.data = get(prefix + "depthwise_conv_.weight")
        self.dw.bias.data = get(prefix + "depthwise_conv_.bias")
        self.bn.weight.data = get(prefix + "batch_norm_.weight")
        self.bn.bias.data = get(prefix + "batch_norm_.bias")
        self.bn.running_mean.data = get(prefix + "batch_norm_.running_mean")
        self.bn.running_var.data = get(prefix + "batch_norm_.running_var")
        self.pw2.weight.data = get(prefix + "pointwise_conv2_.weight")
        self.pw2.bias.data = get(prefix + "pointwise_conv2_.bias")

    def forward(self, x):
        h = self.norm(x)
        h = h.permute(0, 2, 1)  # (batch, hidden, seq)
        h = self.pw1(h)  # (batch, 2*hidden, seq)
        h = F.glu(h, dim=1)  # (batch, hidden, seq)
        h = self.dw(h)
        h = self.bn(h)
        h = F.silu(h)
        h = self.pw2(h)
        h = h.permute(0, 2, 1)  # (batch, seq, hidden)
        return x + h


class ConformerAttention(nn.Module):
    def __init__(self, prefix, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(512)
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(512, 512)
        self.pos_proj = nn.Linear(512, 512, bias=False)

        self.norm.weight.data = get(prefix + "norm_.weight")
        self.norm.bias.data = get(prefix + "norm_.bias")
        self.q_proj.weight.data = get(prefix + "mha_.q_proj.weight")
        self.q_proj.bias.data = get(prefix + "mha_.q_proj.bias")
        self.k_proj.weight.data = get(prefix + "mha_.k_proj.weight")
        self.k_proj.bias.data = get(prefix + "mha_.k_proj.bias")
        self.v_proj.weight.data = get(prefix + "mha_.v_proj.weight")
        self.v_proj.bias.data = get(prefix + "mha_.v_proj.bias")
        self.out_proj.weight.data = get(prefix + "mha_.out_proj.weight")
        self.out_proj.bias.data = get(prefix + "mha_.out_proj.bias")
        self.pos_proj.weight.data = get(prefix + "pos_proj_.weight")

        self.pos_bias_u = get(prefix + "pos_bias_u_")  # (num_heads * head_dim,) or (num_heads, head_dim)
        self.pos_bias_v = get(prefix + "pos_bias_v_")
        print(f"    pos_bias_u shape: {self.pos_bias_u.shape}")
        print(f"    pos_bias_v shape: {self.pos_bias_v.shape}")

    def forward(self, x, pos_emb):
        h = self.norm(x)
        # Relative position multi-head attention
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        batch, seq_len, d_model = q.shape
        head_dim = d_model // self.num_heads
        scale = 1.0 / (head_dim ** 0.5)

        # Reshape to (batch, heads, seq, head_dim)
        q = q.reshape(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.reshape(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.reshape(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)

        # Position biases
        bias_u = self.pos_bias_u.reshape(1, self.num_heads, 1, head_dim)
        bias_v = self.pos_bias_v.reshape(1, self.num_heads, 1, head_dim)

        # Content score: (Q + bias_u) @ K^T
        content_score = torch.matmul(q + bias_u, k.transpose(-2, -1))

        # Position score
        p = self.pos_proj(pos_emb)  # (2*seq-1, d_model)
        p = p.reshape(1, p.shape[0], self.num_heads, head_dim).transpose(1, 2)
        pos_score = torch.matmul(q + bias_v, p.transpose(-2, -1))
        pos_score = self.rel_shift(pos_score)

        scores = (content_score + pos_score) * scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(batch, seq_len, d_model)
        out = self.out_proj(out)
        return x + out

    @staticmethod
    def rel_shift(x):
        # x: (batch, heads, seq_len, 2*seq_len-1)
        b, h, seq_len, pos_len = x.shape
        # Pad left
        x = F.pad(x, (1, 0))  # (b, h, seq, 2*seq)
        x = x.reshape(b, h, pos_len + 1, seq_len)
        x = x[:, :, 1:]  # (b, h, 2*seq-1, seq)
        x = x.reshape(b, h, seq_len, pos_len)
        return x[:, :, :, :seq_len]


class ConformerBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        p = f"encoder_.layers_.{layer_idx}."
        self.ffn1 = FeedForward(p + "ffn1_.")
        self.attn = ConformerAttention(p + "attn_.")
        self.conv = ConformerConvModule(p + "conv_.")
        self.ffn2 = FeedForward(p + "ffn2_.")
        self.final_norm = nn.LayerNorm(512)
        self.final_norm.weight.data = get(p + "final_norm_.weight")
        self.final_norm.bias.data = get(p + "final_norm_.bias")

    def forward(self, x, pos_emb):
        x = self.ffn1(x)
        x = self.attn(x, pos_emb)
        x = self.conv(x)
        x = self.ffn2(x)
        x = self.final_norm(x)
        return x


# Run single conformer block
block0 = ConformerBlock(0)
block0.eval()

pos_emb = sinusoidal_position_embedding(93, 512)

with torch.no_grad():
    layer0_out = block0(sub_out, pos_emb)

print(f"Layer 0 output: {layer0_out.shape}")
print(f"  Py  range: [{layer0_out.min():.4f}, {layer0_out.max():.4f}] mean={layer0_out.mean():.6f}")

compare("after_layer0", "models/debug_after_layer0.bin", layer0_out)

# CTC decode on layer0 output for sanity
ctc_w = get("ctc_decoder_.proj_.weight")  # (1025, 512, 1)
ctc_b = get("ctc_decoder_.proj_.bias")    # (1025,)
ctc_proj = nn.Conv1d(512, 1025, 1)
ctc_proj.weight.data = ctc_w
ctc_proj.bias.data = ctc_b

# Full encoder (all 17 layers)
print("\n=== Running all 17 layers ===")
x = sub_out.clone()
for i in range(17):
    block_i = ConformerBlock(i)
    block_i.eval()
    with torch.no_grad():
        x = block_i(x, pos_emb)
    if i == 0:
        print(f"  After layer {i}: [{x.min():.4f}, {x.max():.4f}] mean={x.mean():.6f}")

print(f"  Final encoder out: [{x.min():.4f}, {x.max():.4f}] mean={x.mean():.6f}")

# CTC decode
with torch.no_grad():
    logits = ctc_proj(x.transpose(1, 2)).transpose(1, 2)
    log_probs = F.log_softmax(logits, dim=-1)
    preds = log_probs.argmax(dim=-1)

blank_id = 1024
tokens = []
prev = -1
for t in range(93):
    tok = preds[0, t].item()
    if tok != blank_id and tok != prev:
        tokens.append(tok)
    prev = tok

print(f"\nPyTorch CTC tokens ({len(tokens)}): {tokens[:30]}")

# Load vocab for detokenization
vocab = []
with open("models/vocab.txt", "r") as vf:
    for line in vf:
        vocab.append(line.strip())

text = ""
for t in tokens:
    if t < len(vocab):
        piece = vocab[t]
        if piece.startswith("\u2581"):  # ▁
            text += " " + piece[1:]
        else:
            text += piece
    else:
        text += f"[{t}]"
print(f"PyTorch text: {text.strip()}")
