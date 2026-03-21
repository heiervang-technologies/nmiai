"""
MarkusNet-860M submission: ONNX YOLO detection + pruned Qwen3.5-0.8B classification.

The classifier is a pruned Qwen3.5-VLM (12 vision ViT blocks + merger + 12 hybrid
language model blocks) with a classification head trained on 22.7k grocery product crops.

Architecture reproduced in pure PyTorch (no transformers dependency at runtime).

Usage:
    python run_markusnet.py --input /data/images --output /predictions.json
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ============================================================================
# Constants
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
YOLO_MODEL = "best.onnx"
MARKUSNET_CKPT = "markusnet_351m_nf4.pt"

NUM_CLASSES = 356
CONF_THRESH = 0.001
NMS_IOU_THRESH = 0.45
MAX_DET = 300
CROP_BATCH_SIZE = 64

# Vision config
VIS_HIDDEN = 768
VIS_INTERMEDIATE = 3072
VIS_NUM_HEADS = 12
VIS_DEPTH = 12
VIS_PATCH_SIZE = 16
VIS_TEMPORAL_PATCH = 2
VIS_NUM_POS_EMBED = 2304
VIS_OUT_HIDDEN = 1024
VIS_SPATIAL_MERGE = 2

# Text config
TXT_HIDDEN = 1024
TXT_INTERMEDIATE = 3584
TXT_NUM_HEADS = 8
TXT_NUM_KV_HEADS = 2
TXT_HEAD_DIM = 256
TXT_NUM_LAYERS = 12
TXT_RMS_EPS = 1e-6
TXT_VOCAB_SIZE = 248320

# Linear attention config
LIN_NUM_K_HEADS = 16
LIN_NUM_V_HEADS = 16
LIN_K_HEAD_DIM = 128
LIN_V_HEAD_DIM = 128
LIN_CONV_KERNEL = 4

LAYER_TYPES = [
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
]

# Special tokens
IMAGE_TOKEN_ID = 248056
VISION_START_TOKEN_ID = 248053
VISION_END_TOKEN_ID = 248054
IM_START_TOKEN_ID = 248045
IM_END_TOKEN_ID = 248046
USER_TOKEN_ID = 846
NEWLINE_TOKEN_ID = 198
CLASSIFY_TOKEN_ID = 91037

# Chat template token sequence (prefix before image tokens, suffix after)
# <|im_start|>user\n<|vision_start|>
CHAT_PREFIX_IDS = [IM_START_TOKEN_ID, USER_TOKEN_ID, NEWLINE_TOKEN_ID, VISION_START_TOKEN_ID]
# <|vision_end|>classify<|im_end|>\n
CHAT_SUFFIX_IDS = [VISION_END_TOKEN_ID, CLASSIFY_TOKEN_ID, IM_END_TOKEN_ID, NEWLINE_TOKEN_ID]

# Image preprocessing for Qwen3.5 processor (match processor defaults)
# factor = patch_size * merge_size = 16 * 2 = 32
QWEN_IMAGE_FACTOR = 32
QWEN_MIN_PIXELS = 65536
QWEN_MAX_PIXELS = 16777216
QWEN_MEAN = [0.5, 0.5, 0.5]
QWEN_STD = [0.5, 0.5, 0.5]

# ROPE config
ROPE_THETA = 10000000
PARTIAL_ROTARY_FACTOR = 0.25
MROPE_SECTION = [11, 11, 10]


# ============================================================================
# YOLO Detection (reused from existing pipeline)
# ============================================================================

def load_onnx_session(model_path: Path, providers: list):
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_info = session.get_inputs()[0]
    return session, input_info.name, tuple(input_info.shape)


def letterbox(image, new_shape, color=(114, 114, 114)):
    height, width = image.shape[:2]
    ratio = min(new_shape[0] / height, new_shape[1] / width)
    resized_wh = (int(round(width * ratio)), int(round(height * ratio)))
    pad_w = (new_shape[1] - resized_wh[0]) / 2
    pad_h = (new_shape[0] - resized_wh[1]) / 2
    if (width, height) != resized_wh:
        image = cv2.resize(image, resized_wh, interpolation=cv2.INTER_LINEAR)
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    bordered = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return bordered, ratio, (pad_w, pad_h)


def preprocess_image(image_bgr, input_shape):
    target_h, target_w = int(input_shape[2]), int(input_shape[3])
    letterboxed, ratio, pad = letterbox(image_bgr, (target_h, target_w))
    image_rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    tensor = image_rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
    return tensor, ratio, pad


def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + areas - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return boxes, scores
    order = np.argsort(scores)[::-1]
    kept_boxes, kept_scores = [], []
    while len(order) > 0:
        current = order[0]
        kept_boxes.append(boxes[current])
        kept_scores.append(scores[current])
        if len(order) == 1:
            break
        iou = compute_iou(boxes[current], boxes[order[1:]])
        order = order[1:][iou <= iou_thresh]
    return np.asarray(kept_boxes, dtype=np.float32), np.asarray(kept_scores, dtype=np.float32)


def decode_yolo_output(output, ratio, pad, original_shape):
    prediction = output[0]
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.T

    box_cxcywh = prediction[:, :4]
    class_scores = prediction[:, 4:]
    scores = np.max(class_scores, axis=1)

    mask = scores > CONF_THRESH
    box_cxcywh = box_cxcywh[mask]
    scores = scores[mask]
    if len(box_cxcywh) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)

    boxes = np.empty_like(box_cxcywh)
    boxes[:, 0] = box_cxcywh[:, 0] - box_cxcywh[:, 2] / 2
    boxes[:, 1] = box_cxcywh[:, 1] - box_cxcywh[:, 3] / 2
    boxes[:, 2] = box_cxcywh[:, 0] + box_cxcywh[:, 2] / 2
    boxes[:, 3] = box_cxcywh[:, 1] + box_cxcywh[:, 3] / 2

    pad_w, pad_h = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / ratio

    original_h, original_w = original_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h)

    valid = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
    boxes = boxes[valid]
    scores = scores[valid]
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)

    boxes, scores = nms(boxes, scores, NMS_IOU_THRESH)
    if len(scores) > MAX_DET:
        top = np.argsort(scores)[::-1][:MAX_DET]
        boxes = boxes[top]
        scores = scores[top]
    return boxes.astype(np.float32), scores.astype(np.float32)


# ============================================================================
# MarkusNet: Pure PyTorch Qwen3.5-VLM (pruned) + Classification Head
# ============================================================================

# --- Utility functions ---

def rms_norm(x, weight, eps=1e-6):
    """RMSNorm with (1 + weight) scaling as used in Qwen3.5."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normed = x.float() * torch.rsqrt(variance + eps)
    return (normed * (1.0 + weight.float())).to(x.dtype)


def rms_norm_gated(x, weight, gate, eps=1e-6):
    """Gated RMSNorm: norm(x) * silu(gate)."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normed = x.float() * torch.rsqrt(variance + eps)
    normed = (weight * normed.to(x.dtype))
    normed = normed * F.silu(gate.float())
    return normed.to(x.dtype)


def layer_norm(x, weight, bias, eps=1e-6):
    """Standard LayerNorm."""
    return F.layer_norm(x, weight.shape, weight, bias, eps)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


# --- Vision Encoder ---

class VisionEncoder(nn.Module):
    """Qwen3.5 Vision Encoder: ViT with 3D patch embed + rotary pos emb + merger."""

    def __init__(self):
        super().__init__()
        # Patch embedding: 3D conv
        self.patch_embed_proj = nn.Conv3d(
            3, VIS_HIDDEN,
            kernel_size=[VIS_TEMPORAL_PATCH, VIS_PATCH_SIZE, VIS_PATCH_SIZE],
            stride=[VIS_TEMPORAL_PATCH, VIS_PATCH_SIZE, VIS_PATCH_SIZE],
            bias=True,
        )
        # Position embedding
        self.pos_embed = nn.Embedding(VIS_NUM_POS_EMBED, VIS_HIDDEN)
        self.num_grid_per_side = int(VIS_NUM_POS_EMBED ** 0.5)  # 48

        # Rotary
        head_dim = VIS_HIDDEN // VIS_NUM_HEADS  # 64
        rot_dim = head_dim // 2  # 32
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rot_dim, 2, dtype=torch.float) / rot_dim))
        self.register_buffer("rot_inv_freq", inv_freq)

        # Blocks
        self.block_norm1_w = nn.ParameterList()
        self.block_norm1_b = nn.ParameterList()
        self.block_norm2_w = nn.ParameterList()
        self.block_norm2_b = nn.ParameterList()
        self.block_attn_qkv_w = nn.ParameterList()
        self.block_attn_qkv_b = nn.ParameterList()
        self.block_attn_proj_w = nn.ParameterList()
        self.block_attn_proj_b = nn.ParameterList()
        self.block_mlp_fc1_w = nn.ParameterList()
        self.block_mlp_fc1_b = nn.ParameterList()
        self.block_mlp_fc2_w = nn.ParameterList()
        self.block_mlp_fc2_b = nn.ParameterList()

        for _ in range(VIS_DEPTH):
            self.block_norm1_w.append(nn.Parameter(torch.zeros(VIS_HIDDEN)))
            self.block_norm1_b.append(nn.Parameter(torch.zeros(VIS_HIDDEN)))
            self.block_norm2_w.append(nn.Parameter(torch.zeros(VIS_HIDDEN)))
            self.block_norm2_b.append(nn.Parameter(torch.zeros(VIS_HIDDEN)))
            self.block_attn_qkv_w.append(nn.Parameter(torch.zeros(VIS_HIDDEN * 3, VIS_HIDDEN)))
            self.block_attn_qkv_b.append(nn.Parameter(torch.zeros(VIS_HIDDEN * 3)))
            self.block_attn_proj_w.append(nn.Parameter(torch.zeros(VIS_HIDDEN, VIS_HIDDEN)))
            self.block_attn_proj_b.append(nn.Parameter(torch.zeros(VIS_HIDDEN)))
            self.block_mlp_fc1_w.append(nn.Parameter(torch.zeros(VIS_INTERMEDIATE, VIS_HIDDEN)))
            self.block_mlp_fc1_b.append(nn.Parameter(torch.zeros(VIS_INTERMEDIATE)))
            self.block_mlp_fc2_w.append(nn.Parameter(torch.zeros(VIS_HIDDEN, VIS_INTERMEDIATE)))
            self.block_mlp_fc2_b.append(nn.Parameter(torch.zeros(VIS_HIDDEN)))

        # Merger
        merge_hidden = VIS_HIDDEN * (VIS_SPATIAL_MERGE ** 2)  # 768 * 4 = 3072
        self.merger_norm_w = nn.Parameter(torch.zeros(VIS_HIDDEN))
        self.merger_norm_b = nn.Parameter(torch.zeros(VIS_HIDDEN))
        self.merger_fc1_w = nn.Parameter(torch.zeros(merge_hidden, merge_hidden))
        self.merger_fc1_b = nn.Parameter(torch.zeros(merge_hidden))
        self.merger_fc2_w = nn.Parameter(torch.zeros(VIS_OUT_HIDDEN, merge_hidden))
        self.merger_fc2_b = nn.Parameter(torch.zeros(VIS_OUT_HIDDEN))

    def _compute_rotary(self, grid_thw):
        """Compute rotary position embeddings for vision tokens."""
        merge_size = VIS_SPATIAL_MERGE
        max_hw = max(grid_thw[0][1], grid_thw[0][2])
        seq = torch.arange(max_hw, device=self.rot_inv_freq.device, dtype=self.rot_inv_freq.dtype)
        freqs = torch.outer(seq, self.rot_inv_freq)  # (max_hw, rot_dim/2)

        t, h, w = grid_thw[0]
        merged_h, merged_w = h // merge_size, w // merge_size

        block_rows = torch.arange(merged_h, device=freqs.device)
        block_cols = torch.arange(merged_w, device=freqs.device)
        intra_row = torch.arange(merge_size, device=freqs.device)
        intra_col = torch.arange(merge_size, device=freqs.device)

        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        pos_ids = torch.stack((row_idx, col_idx), dim=-1)
        if t > 1:
            pos_ids = pos_ids.repeat(t, 1)

        embeddings = freqs[pos_ids]  # (num_tokens, 2, rot_dim/2)
        embeddings = embeddings.flatten(1)  # (num_tokens, rot_dim)
        return embeddings

    def _compute_pos_embed(self, grid_thw):
        """Compute interpolated position embeddings."""
        t, h, w = grid_thw[0]
        device = self.pos_embed.weight.device
        merge_size = VIS_SPATIAL_MERGE

        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        h_floor = h_idxs.int()
        w_floor = w_idxs.int()
        h_ceil = (h_floor + 1).clip(max=self.num_grid_per_side - 1)
        w_ceil = (w_floor + 1).clip(max=self.num_grid_per_side - 1)

        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        base_h = h_floor * self.num_grid_per_side
        base_h_ceil = h_ceil * self.num_grid_per_side

        indices = [
            (base_h[None].T + w_floor[None]).flatten(),
            (base_h[None].T + w_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_floor[None]).flatten(),
            (base_h_ceil[None].T + w_ceil[None]).flatten(),
        ]
        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        idx_tensor = torch.tensor([i.tolist() for i in indices], dtype=torch.long, device=device)
        weight_tensor = torch.tensor([w.tolist() for w in weights], dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        # Repeat for temporal dim and permute for merge
        patch_pos = patch_pos.repeat(t, 1)
        patch_pos = (
            patch_pos.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        return patch_pos

    def _vision_attention(self, x, cos, sin, qkv_w, qkv_b, proj_w, proj_b):
        """Single vision attention block (no causal mask, single sequence)."""
        seq_len = x.shape[0]
        head_dim = VIS_HIDDEN // VIS_NUM_HEADS

        qkv = F.linear(x, qkv_w, qkv_b)  # (seq_len, 3*hidden)
        qkv = qkv.reshape(seq_len, 3, VIS_NUM_HEADS, head_dim).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (seq_len, num_heads, head_dim)

        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Standard attention: (1, num_heads, seq_len, head_dim)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        scale = head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)

        out = out.squeeze(0).transpose(0, 1).reshape(seq_len, -1).contiguous()
        return F.linear(out, proj_w, proj_b)

    def forward(self, pixel_values, grid_thw):
        """
        pixel_values: (num_patches, C * temporal * patch_h * patch_w) flattened patches
        grid_thw: list of (t, h, w) tuples
        Returns: merged features (num_merged_tokens, VIS_OUT_HIDDEN)
        """
        # Cast input to match model dtype (half/bfloat16 on CUDA, float on CPU)
        pixel_values = pixel_values.to(self.patch_embed_proj.weight.dtype)

        # Patch embed
        x = pixel_values.view(-1, 3, VIS_TEMPORAL_PATCH, VIS_PATCH_SIZE, VIS_PATCH_SIZE)
        x = self.patch_embed_proj(x).view(-1, VIS_HIDDEN)

        # Position embedding
        pos_embed = self._compute_pos_embed(grid_thw)
        x = x + pos_embed

        # Rotary
        rotary = self._compute_rotary(grid_thw)
        emb = torch.cat((rotary, rotary), dim=-1)
        cos, sin = emb.cos(), emb.sin()

        # Vision transformer blocks
        for i in range(VIS_DEPTH):
            # Attention
            normed = layer_norm(x, self.block_norm1_w[i], self.block_norm1_b[i])
            attn_out = self._vision_attention(
                normed, cos, sin,
                self.block_attn_qkv_w[i], self.block_attn_qkv_b[i],
                self.block_attn_proj_w[i], self.block_attn_proj_b[i],
            )
            x = x + attn_out

            # MLP
            normed = layer_norm(x, self.block_norm2_w[i], self.block_norm2_b[i])
            mlp_out = F.linear(normed, self.block_mlp_fc1_w[i], self.block_mlp_fc1_b[i])
            mlp_out = F.gelu(mlp_out, approximate="tanh")
            mlp_out = F.linear(mlp_out, self.block_mlp_fc2_w[i], self.block_mlp_fc2_b[i])
            x = x + mlp_out

        # Merger: norm -> reshape -> fc1 -> gelu -> fc2
        x = layer_norm(x, self.merger_norm_w, self.merger_norm_b)
        # Reshape: group spatial_merge_size^2 tokens together
        merge_hidden = VIS_HIDDEN * (VIS_SPATIAL_MERGE ** 2)
        x = x.view(-1, merge_hidden)
        x = F.linear(x, self.merger_fc1_w, self.merger_fc1_b)
        x = F.gelu(x)
        x = F.linear(x, self.merger_fc2_w, self.merger_fc2_b)
        return x


# --- Language Model ---

class LanguageModel(nn.Module):
    """Qwen3.5 Text Model: hybrid Mamba/attention layers."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(TXT_VOCAB_SIZE, TXT_HIDDEN)

        # Per-layer parameters stored as ParameterDicts
        self.layers = nn.ModuleList()
        for i in range(TXT_NUM_LAYERS):
            self.layers.append(DecoderLayer(LAYER_TYPES[i], i))

        # Final norm
        self.norm_weight = nn.Parameter(torch.zeros(TXT_HIDDEN))

    def forward(self, inputs_embeds, position_ids=None, attention_mask=None):
        """
        inputs_embeds: (B, seq_len, hidden)
        position_ids: (3, B, seq_len) for mRoPE
        Returns: (B, seq_len, hidden)
        """
        hidden = inputs_embeds
        B, seq_len, _ = hidden.shape

        # Compute RoPE embeddings for full attention layers
        cos, sin = None, None
        if position_ids is not None:
            cos, sin = self._compute_rope(hidden, position_ids)

        # Compute causal mask for full attention layers
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=hidden.device, dtype=hidden.dtype),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        for layer in self.layers:
            hidden = layer(hidden, cos, sin, causal_mask, attention_mask)

        hidden = rms_norm(hidden, self.norm_weight, TXT_RMS_EPS)
        return hidden

    def _compute_rope(self, x, position_ids):
        """Compute multimodal RoPE (mRoPE) embeddings."""
        # head_dim = 256, partial_rotary = 0.25, so rotary_dim = 64
        head_dim = TXT_HEAD_DIM
        rotary_dim = int(head_dim * PARTIAL_ROTARY_FACTOR)  # 64

        inv_freq = 1.0 / (ROPE_THETA ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float, device=x.device) / rotary_dim
        ))

        # position_ids: (3, B, seq_len)
        inv_freq_expanded = inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        # freqs: (3, B, seq_len, rotary_dim/2)

        # Apply interleaved mRoPE
        freqs_t = freqs[0].clone()
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = MROPE_SECTION[dim_idx] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim_idx, ..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin


class DecoderLayer(nn.Module):
    def __init__(self, layer_type, layer_idx):
        super().__init__()
        self.layer_type = layer_type
        self.layer_idx = layer_idx

        self.input_layernorm_weight = nn.Parameter(torch.zeros(TXT_HIDDEN))
        self.post_attn_layernorm_weight = nn.Parameter(torch.zeros(TXT_HIDDEN))

        # MLP
        self.mlp_gate_proj = nn.Linear(TXT_HIDDEN, TXT_INTERMEDIATE, bias=False)
        self.mlp_up_proj = nn.Linear(TXT_HIDDEN, TXT_INTERMEDIATE, bias=False)
        self.mlp_down_proj = nn.Linear(TXT_INTERMEDIATE, TXT_HIDDEN, bias=False)

        if layer_type == "full_attention":
            # q_proj outputs num_heads * head_dim * 2 (for gate)
            self.q_proj = nn.Linear(TXT_HIDDEN, TXT_NUM_HEADS * TXT_HEAD_DIM * 2, bias=False)
            self.k_proj = nn.Linear(TXT_HIDDEN, TXT_NUM_KV_HEADS * TXT_HEAD_DIM, bias=False)
            self.v_proj = nn.Linear(TXT_HIDDEN, TXT_NUM_KV_HEADS * TXT_HEAD_DIM, bias=False)
            self.o_proj = nn.Linear(TXT_NUM_HEADS * TXT_HEAD_DIM, TXT_HIDDEN, bias=False)
            self.q_norm_weight = nn.Parameter(torch.zeros(TXT_HEAD_DIM))
            self.k_norm_weight = nn.Parameter(torch.zeros(TXT_HEAD_DIM))
        else:
            # Linear attention (Gated DeltaNet)
            key_dim = LIN_K_HEAD_DIM * LIN_NUM_K_HEADS  # 2048
            value_dim = LIN_V_HEAD_DIM * LIN_NUM_V_HEADS  # 2048
            conv_dim = key_dim * 2 + value_dim  # 6144

            self.in_proj_qkv = nn.Linear(TXT_HIDDEN, conv_dim, bias=False)
            self.in_proj_z = nn.Linear(TXT_HIDDEN, value_dim, bias=False)
            self.in_proj_b = nn.Linear(TXT_HIDDEN, LIN_NUM_V_HEADS, bias=False)
            self.in_proj_a = nn.Linear(TXT_HIDDEN, LIN_NUM_V_HEADS, bias=False)

            self.conv1d = nn.Conv1d(
                conv_dim, conv_dim, kernel_size=LIN_CONV_KERNEL,
                groups=conv_dim, bias=False, padding=LIN_CONV_KERNEL - 1,
            )

            self.dt_bias = nn.Parameter(torch.ones(LIN_NUM_V_HEADS))
            self.A_log = nn.Parameter(torch.zeros(LIN_NUM_V_HEADS))

            self.gated_norm_weight = nn.Parameter(torch.ones(LIN_V_HEAD_DIM))
            self.out_proj = nn.Linear(value_dim, TXT_HIDDEN, bias=False)

    def forward(self, hidden, cos, sin, causal_mask, attention_mask):
        residual = hidden
        hidden = rms_norm(hidden, self.input_layernorm_weight, TXT_RMS_EPS)

        if self.layer_type == "full_attention":
            hidden = self._full_attention(hidden, cos, sin, causal_mask)
        else:
            hidden = self._linear_attention(hidden, attention_mask)

        hidden = residual + hidden

        residual = hidden
        hidden = rms_norm(hidden, self.post_attn_layernorm_weight, TXT_RMS_EPS)
        hidden = self.mlp_down_proj(F.silu(self.mlp_gate_proj(hidden)) * self.mlp_up_proj(hidden))
        hidden = residual + hidden
        return hidden

    def _full_attention(self, hidden, cos, sin, causal_mask):
        B, seq_len, _ = hidden.shape
        head_dim = TXT_HEAD_DIM

        # Q with gate
        qg = self.q_proj(hidden)
        qg = qg.view(B, seq_len, -1, head_dim * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(B, seq_len, -1)

        # Q/K norm
        q = q.reshape(B, seq_len, TXT_NUM_HEADS, head_dim)
        q = rms_norm(q, self.q_norm_weight, TXT_RMS_EPS)
        q = q.transpose(1, 2)  # (B, num_heads, seq_len, head_dim)

        k = self.k_proj(hidden).view(B, seq_len, TXT_NUM_KV_HEADS, head_dim)
        k = rms_norm(k, self.k_norm_weight, TXT_RMS_EPS)
        k = k.transpose(1, 2)

        v = self.v_proj(hidden).view(B, seq_len, TXT_NUM_KV_HEADS, head_dim).transpose(1, 2)

        # Apply partial rotary embeddings
        if cos is not None and sin is not None:
            rotary_dim = cos.shape[-1]
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

            cos_ = cos.unsqueeze(1)  # (B, 1, seq_len, rotary_dim)
            sin_ = sin.unsqueeze(1)

            q_rot = (q_rot * cos_) + (rotate_half(q_rot) * sin_)
            k_rot = (k_rot * cos_) + (rotate_half(k_rot) * sin_)

            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

        # GQA: repeat k,v
        num_kv_groups = TXT_NUM_HEADS // TXT_NUM_KV_HEADS
        if num_kv_groups > 1:
            k = k[:, :, None, :, :].expand(B, TXT_NUM_KV_HEADS, num_kv_groups, seq_len, head_dim)
            k = k.reshape(B, TXT_NUM_HEADS, seq_len, head_dim)
            v = v[:, :, None, :, :].expand(B, TXT_NUM_KV_HEADS, num_kv_groups, seq_len, head_dim)
            v = v.reshape(B, TXT_NUM_HEADS, seq_len, head_dim)

        scale = head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal_mask is not None:
            attn = attn + causal_mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, seq_len, -1).contiguous()
        # Apply output gate
        out = out * torch.sigmoid(gate)
        out = self.o_proj(out)
        return out

    def _linear_attention(self, hidden, attention_mask):
        """Gated DeltaNet linear attention (torch slow path)."""
        B, seq_len, _ = hidden.shape
        key_dim = LIN_K_HEAD_DIM * LIN_NUM_K_HEADS
        value_dim = LIN_V_HEAD_DIM * LIN_NUM_V_HEADS

        mixed_qkv = self.in_proj_qkv(hidden)  # (B, seq_len, conv_dim)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # (B, conv_dim, seq_len)

        z = self.in_proj_z(hidden)  # (B, seq_len, value_dim)
        z = z.reshape(B, seq_len, -1, LIN_V_HEAD_DIM)

        b = self.in_proj_b(hidden)  # (B, seq_len, num_v_heads)
        a = self.in_proj_a(hidden)  # (B, seq_len, num_v_heads)

        # Causal conv1d (slow path)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)  # (B, seq_len, conv_dim)

        query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)

        query = query.reshape(B, seq_len, -1, LIN_K_HEAD_DIM)
        key = key.reshape(B, seq_len, -1, LIN_K_HEAD_DIM)
        value = value.reshape(B, seq_len, -1, LIN_V_HEAD_DIM)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # num_k_heads == num_v_heads, no repeat needed

        # Chunk gated delta rule (torch slow path)
        core_out, _ = torch_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        # Gated RMSNorm
        core_out = core_out.reshape(-1, LIN_V_HEAD_DIM)
        z_flat = z.reshape(-1, LIN_V_HEAD_DIM)
        core_out = rms_norm_gated(core_out, self.gated_norm_weight, z_flat, TXT_RMS_EPS)
        core_out = core_out.reshape(B, seq_len, -1)

        return self.out_proj(core_out)


def torch_chunk_gated_delta_rule(
    query, key, value, g, beta,
    chunk_size=64, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Torch implementation of chunked gated delta rule."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_chunk = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_chunk @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# --- Classification Head ---

class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(TXT_HIDDEN, TXT_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.0),  # No dropout at inference
            nn.Linear(TXT_HIDDEN, NUM_CLASSES),
        )

    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)
        return self.head(pooled)


# --- Full MarkusNet ---

class MarkusNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = VisionEncoder()
        self.language = LanguageModel()
        self.cls_head = ClassificationHead()

    def load_checkpoint(self, ckpt_path, device):
        NF4_TABLE = torch.tensor([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
        ], dtype=torch.float32, device=device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_state = {}

        # Dequantize NF4 tensors
        for k, q in ckpt['nf4_state'].items():
            packed = q['packed'].to(device)
            scales = q['scales'].to(device)
            high = (packed >> 4).long()
            low = (packed & 0x0F).long()
            indices = torch.stack([high, low], dim=1).reshape(-1)
            values = NF4_TABLE[indices].reshape(-1, 64) * scales.float().unsqueeze(1)
            model_state[k] = values.reshape(-1)[:q['numel']].reshape(q['shape']).half()

        # Add FP16 tensors
        for k, v in ckpt['fp16_state'].items():
            model_state[k] = v.to(device)

        cls_state = ckpt['cls_head_state']

        has_embed_tokens = "model.language_model.embed_tokens.weight" in model_state
        # Load vision encoder weights
        self._load_vision(model_state)
        # Load language model weights
        self._load_language(model_state)
        if not has_embed_tokens:
            self._load_token_embeddings(ckpt, device)
        # Load classification head
        self.cls_head.head[0].weight.data = cls_state["head.0.weight"].to(device)
        self.cls_head.head[0].bias.data = cls_state["head.0.bias"].to(device)
        self.cls_head.head[3].weight.data = cls_state["head.3.weight"].to(device)
        self.cls_head.head[3].bias.data = cls_state["head.3.bias"].to(device)

    def _load_token_embeddings(self, ckpt, device):
        token_ids = ckpt.get("token_ids")
        token_embeds = ckpt.get("token_embeds")
        if token_ids is None or token_embeds is None:
            raise RuntimeError(
                "Checkpoint is missing embed_tokens and required token embeddings. "
                "Re-export NF4 with token_ids/token_embeds metadata."
            )

        embed_device = self.language.embed_tokens.weight.device
        token_ids = torch.as_tensor(token_ids, dtype=torch.long, device=embed_device)
        token_embeds = token_embeds.to(device=embed_device, dtype=self.language.embed_tokens.weight.dtype)

        if token_embeds.ndim != 2 or token_embeds.shape[0] != token_ids.numel() or token_embeds.shape[1] != TXT_HIDDEN:
            raise RuntimeError(
                f"Invalid token embedding payload: ids={token_ids.shape}, embeds={token_embeds.shape}"
            )

        self.language.embed_tokens.weight.data.zero_()
        self.language.embed_tokens.weight.data[token_ids] = token_embeds

    def _load_vision(self, state):
        prefix = "model.visual."
        self.vision.patch_embed_proj.weight.data = state[prefix + "patch_embed.proj.weight"]
        self.vision.patch_embed_proj.bias.data = state[prefix + "patch_embed.proj.bias"]
        self.vision.pos_embed.weight.data = state[prefix + "pos_embed.weight"]

        for i in range(VIS_DEPTH):
            bp = f"{prefix}blocks.{i}."
            self.vision.block_norm1_w[i].data = state[bp + "norm1.weight"]
            self.vision.block_norm1_b[i].data = state[bp + "norm1.bias"]
            self.vision.block_norm2_w[i].data = state[bp + "norm2.weight"]
            self.vision.block_norm2_b[i].data = state[bp + "norm2.bias"]
            self.vision.block_attn_qkv_w[i].data = state[bp + "attn.qkv.weight"]
            self.vision.block_attn_qkv_b[i].data = state[bp + "attn.qkv.bias"]
            self.vision.block_attn_proj_w[i].data = state[bp + "attn.proj.weight"]
            self.vision.block_attn_proj_b[i].data = state[bp + "attn.proj.bias"]
            self.vision.block_mlp_fc1_w[i].data = state[bp + "mlp.linear_fc1.weight"]
            self.vision.block_mlp_fc1_b[i].data = state[bp + "mlp.linear_fc1.bias"]
            self.vision.block_mlp_fc2_w[i].data = state[bp + "mlp.linear_fc2.weight"]
            self.vision.block_mlp_fc2_b[i].data = state[bp + "mlp.linear_fc2.bias"]

        mp = prefix + "merger."
        self.vision.merger_norm_w.data = state[mp + "norm.weight"]
        self.vision.merger_norm_b.data = state[mp + "norm.bias"]
        self.vision.merger_fc1_w.data = state[mp + "linear_fc1.weight"]
        self.vision.merger_fc1_b.data = state[mp + "linear_fc1.bias"]
        self.vision.merger_fc2_w.data = state[mp + "linear_fc2.weight"]
        self.vision.merger_fc2_b.data = state[mp + "linear_fc2.bias"]

    def _load_language(self, state):
        prefix = "model.language_model."
        # embed_tokens and norm may be absent in stripped/exported checkpoints
        if prefix + "embed_tokens.weight" in state:
            self.language.embed_tokens.weight.data = state[prefix + "embed_tokens.weight"]
        if prefix + "norm.weight" in state:
            self.language.norm_weight.data = state[prefix + "norm.weight"]

        for i in range(TXT_NUM_LAYERS):
            lp = f"{prefix}layers.{i}."
            layer = self.language.layers[i]
            layer.input_layernorm_weight.data = state[lp + "input_layernorm.weight"]
            layer.post_attn_layernorm_weight.data = state[lp + "post_attention_layernorm.weight"]

            layer.mlp_gate_proj.weight.data = state[lp + "mlp.gate_proj.weight"]
            layer.mlp_up_proj.weight.data = state[lp + "mlp.up_proj.weight"]
            layer.mlp_down_proj.weight.data = state[lp + "mlp.down_proj.weight"]

            if LAYER_TYPES[i] == "full_attention":
                layer.q_proj.weight.data = state[lp + "self_attn.q_proj.weight"]
                layer.k_proj.weight.data = state[lp + "self_attn.k_proj.weight"]
                layer.v_proj.weight.data = state[lp + "self_attn.v_proj.weight"]
                layer.o_proj.weight.data = state[lp + "self_attn.o_proj.weight"]
                layer.q_norm_weight.data = state[lp + "self_attn.q_norm.weight"]
                layer.k_norm_weight.data = state[lp + "self_attn.k_norm.weight"]
            else:
                layer.in_proj_qkv.weight.data = state[lp + "linear_attn.in_proj_qkv.weight"]
                layer.in_proj_z.weight.data = state[lp + "linear_attn.in_proj_z.weight"]
                layer.in_proj_b.weight.data = state[lp + "linear_attn.in_proj_b.weight"]
                layer.in_proj_a.weight.data = state[lp + "linear_attn.in_proj_a.weight"]
                layer.conv1d.weight.data = state[lp + "linear_attn.conv1d.weight"]
                layer.dt_bias.data = state[lp + "linear_attn.dt_bias"]
                layer.A_log.data = state[lp + "linear_attn.A_log"]
                layer.gated_norm_weight.data = state[lp + "linear_attn.norm.weight"]
                layer.out_proj.weight.data = state[lp + "linear_attn.out_proj.weight"]

    @torch.inference_mode()
    def classify_crops(self, crops_pil, device):
        """
        Classify a batch of PIL image crops.
        Constructs the full chat-template token sequence matching training.
        Returns: (category_ids, confidences) as numpy arrays
        """
        if not crops_pil:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        all_cat_ids = []
        all_confs = []

        for start in range(0, len(crops_pil), CROP_BATCH_SIZE):
            batch_crops = crops_pil[start:start + CROP_BATCH_SIZE]

            batch_logits = []
            for crop in batch_crops:
                pv, grid_thw = self._preprocess_image(crop, device)
                # Run vision encoder
                vis_embeds = self.vision(pv, [grid_thw])  # (num_merged_tokens, 1024)

                num_image_tokens = vis_embeds.shape[0]
                # t_patches, h_patches, w_patches
                t, h, w = grid_thw
                llm_grid_h = h // VIS_SPATIAL_MERGE
                llm_grid_w = w // VIS_SPATIAL_MERGE

                # Build input_ids: prefix + image_placeholders + suffix
                prefix_ids = CHAT_PREFIX_IDS
                suffix_ids = CHAT_SUFFIX_IDS
                image_ids = [IMAGE_TOKEN_ID] * num_image_tokens
                input_ids = torch.tensor(
                    [prefix_ids + image_ids + suffix_ids],
                    dtype=torch.long, device=device,
                )

                # Get text embeddings
                inputs_embeds = self.language.embed_tokens(input_ids)  # (1, seq_len, hidden)

                # Scatter vision embeddings into image token positions
                image_mask = (input_ids == IMAGE_TOKEN_ID)  # (1, seq_len)
                image_mask_3d = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask_3d,
                    vis_embeds.to(inputs_embeds.dtype),
                )

                # Build 3D position IDs for mRoPE
                # Text tokens get sequential IDs, image tokens get spatial grid IDs
                seq_len = input_ids.shape[1]
                position_ids = self._build_position_ids(
                    len(prefix_ids), num_image_tokens, len(suffix_ids),
                    t, llm_grid_h, llm_grid_w, device,
                )

                # Forward through language model
                hidden = self.language(inputs_embeds, position_ids=position_ids)
                logits = self.cls_head(hidden)
                batch_logits.append(logits)

            logits = torch.cat(batch_logits, dim=0)
            probs = F.softmax(logits, dim=-1)
            conf, cat_ids = probs.max(dim=-1)

            all_cat_ids.extend(cat_ids.cpu().tolist())
            all_confs.extend(conf.cpu().tolist())

        return np.asarray(all_cat_ids, dtype=np.int64), np.asarray(all_confs, dtype=np.float32)

    def _build_position_ids(self, prefix_len, num_img_tokens, suffix_len,
                            t, llm_grid_h, llm_grid_w, device):
        """Build mRoPE 3D position IDs matching Qwen3.5 get_rope_index."""
        total_len = prefix_len + num_img_tokens + suffix_len

        # Text prefix: sequential positions
        st_idx = 0
        prefix_pos = torch.arange(prefix_len, device=device).unsqueeze(0).expand(3, -1) + st_idx

        # Image tokens: spatial grid positions
        st_idx = prefix_len
        t_index = torch.arange(t, device=device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
        h_index = torch.arange(llm_grid_h, device=device).view(1, -1, 1).expand(t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w, device=device).view(1, 1, -1).expand(t, llm_grid_h, -1).flatten()
        image_pos = torch.stack([t_index, h_index, w_index]) + st_idx  # (3, num_img_tokens)

        # Text suffix: sequential from after image
        st_idx = image_pos.max() + 1
        suffix_pos = torch.arange(suffix_len, device=device).unsqueeze(0).expand(3, -1) + st_idx

        # Concatenate
        position_ids = torch.cat([prefix_pos, image_pos, suffix_pos], dim=1)
        return position_ids.unsqueeze(1)  # (3, 1, seq_len)

    def smart_resize(self, height, width, factor=QWEN_IMAGE_FACTOR, min_pixels=QWEN_MIN_PIXELS, max_pixels=QWEN_MAX_PIXELS):
        """Match transformers Qwen smart_resize: keep aspect ratio and divisibility by factor."""
        if max(height, width) / max(1, min(height, width)) > 200:
            return factor, factor
        h_bar = max(factor, round(height / factor) * factor)
        w_bar = max(factor, round(width / factor) * factor)
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def _preprocess_image(self, pil_img, device):
        """
        Preprocess a PIL image into pixel_values patches for the vision encoder.
        Uses Qwen smart resize + normalization.

        Returns: (pixel_values tensor, (t, h_patches, w_patches) tuple)
        """
        img = pil_img.convert("RGB")
        orig_w, orig_h = img.size
        target_h, target_w = self.smart_resize(orig_h, orig_w)
        img = img.resize((target_w, target_h), Image.BICUBIC)

        # Convert to tensor and normalize
        img_np = np.array(img, dtype=np.float32) / 255.0
        for c in range(3):
            img_np[:, :, c] = (img_np[:, :, c] - QWEN_MEAN[c]) / QWEN_STD[c]

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device)  # (3, H, W)

        # Create patches: temporal_patch=2 means we need 2 frames
        # For a single image, Qwen duplicates it to make 2 temporal frames
        h_patches = target_h // VIS_PATCH_SIZE
        w_patches = target_w // VIS_PATCH_SIZE

        # Stack 2 identical frames
        frames = torch.stack([img_tensor, img_tensor])  # (2, 3, H, W)

        # Reshape into patches: (num_patches, C * temporal * patch_h * patch_w)
        # The conv3d expects (N, C, T, H, W) but we feed flattened patches
        # num_patches = (T / temporal_patch) * (H / patch_size) * (W / patch_size)
        # = 1 * 28 * 28 = 784
        t_patches = 1  # 2 frames / temporal_patch_size 2

        # Reshape: (2, 3, H, W) -> patches of (temporal_patch, patch_h, patch_w)
        # Unfold into patches
        C = 3
        T, H, W = VIS_TEMPORAL_PATCH, target_h, target_w
        pH, pW = VIS_PATCH_SIZE, VIS_PATCH_SIZE
        pT = VIS_TEMPORAL_PATCH

        # (T, C, H, W) -> (T/pT, C, pT, H/pH, pH, W/pW, pW)
        x = frames.reshape(T // pT, pT, C, H // pH, pH, W // pW, pW)
        # -> (T/pT, H/pH, W/pW, C, pT, pH, pW)
        x = x.permute(0, 3, 5, 2, 1, 4, 6)
        # -> (num_patches, C * pT * pH * pW)
        x = x.reshape(-1, C * pT * pH * pW)

        grid_thw = (t_patches, h_patches, w_patches)
        return x, grid_thw


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Load YOLO detector
    available_providers = ort.get_available_providers()
    use_cuda = "CUDAExecutionProvider" in available_providers
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]

    yolo_path = SCRIPT_DIR / YOLO_MODEL
    if not yolo_path.exists():
        # Try alternate names
        for alt in ["yolo11x_v3.onnx", "yolo26x_v3.onnx"]:
            alt_path = SCRIPT_DIR / alt
            if alt_path.exists():
                yolo_path = alt_path
                break

    print(f"Loading YOLO from {yolo_path}")
    yolo_session, yolo_input_name, yolo_input_shape = load_onnx_session(yolo_path, providers)

    # Load MarkusNet
    ckpt_path = SCRIPT_DIR / MARKUSNET_CKPT
    print(f"Loading MarkusNet from {ckpt_path}")
    model = MarkusNet()
    model.load_checkpoint(str(ckpt_path), device)
    model = model.to(dtype).to(device)
    model.eval()
    print("Models loaded.")

    # Process images
    input_dir = Path(args.input)
    output_path = Path(args.output)
    image_paths = sorted(
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg")) +
        list(input_dir.glob("*.png"))
    )

    results = []
    for img_path in image_paths:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue

        # Detect
        tensor, ratio, pad = preprocess_image(image_bgr, yolo_input_shape)
        outputs = yolo_session.run(None, {yolo_input_name: tensor})
        boxes, det_scores = decode_yolo_output(outputs, ratio, pad, image_bgr.shape[:2])

        if len(boxes) == 0:
            continue

        # Crop and classify
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        crops = []
        for box in boxes:
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(image_pil.width, int(box[2]))
            y2 = min(image_pil.height, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                crops.append(Image.new("RGB", (32, 32), (114, 114, 114)))
            else:
                crops.append(image_pil.crop((x1, y1, x2, y2)))

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            cat_ids, cls_conf = model.classify_crops(crops, device)

        # Combine detection and classification scores
        final_scores = np.clip(det_scores * (0.70 + 0.30 * cls_conf), 0.0, 1.0)

        # Category aliases: merge umlaut spelling variants to canonical IDs
        # Maps rare spelling -> common spelling (more training data = better prior)
        ALIASES = {59: 61, 170: 260, 36: 201}

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cat_id = int(cat_ids[idx])
            cat_id = ALIASES.get(cat_id, cat_id)
            results.append({
                "image_id": int(img_path.stem.replace("img_", "")),
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                "category_id": cat_id,
                "score": round(float(final_scores[idx]), 4),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results))
    print(f"Wrote {len(results)} detections to {output_path}")


if __name__ == "__main__":
    main()
