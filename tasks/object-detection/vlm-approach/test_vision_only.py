"""
Test: Can we load and run the Qwen3-VL-2B vision encoder
in a sandbox-like environment using ONLY torch + PIL + numpy?

No transformers, no timm, no ultralytics.
Pure PyTorch ViT implementation.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


class VisionBlock(nn.Module):
    """Standard ViT block: LayerNorm -> Attention -> LayerNorm -> MLP"""
    def __init__(self, hidden=1024, heads=16, mlp_dim=4096):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(hidden, 3 * hidden)
        self.attn.proj = nn.Linear(hidden, hidden)
        self.mlp = nn.Module()
        self.mlp.linear_fc1 = nn.Linear(hidden, mlp_dim)
        self.mlp.linear_fc2 = nn.Linear(mlp_dim, hidden)
        self.heads = heads
        self.head_dim = hidden // heads

    def forward(self, x):
        # Self-attention
        B, N, C = x.shape
        residual = x
        x = self.norm1(x)
        qkv = self.attn.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.attn.proj(x)
        x = residual + x

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp.linear_fc1(x)
        x = F.gelu(x)
        x = self.mlp.linear_fc2(x)
        x = residual + x
        return x


class Qwen3VLVisionEncoder(nn.Module):
    """Pure PyTorch reconstruction of Qwen3-VL-2B vision encoder."""
    def __init__(self, num_blocks=24, hidden=1024, heads=16, mlp_dim=4096,
                 patch_size=16, out_hidden=2048, num_classes=356):
        super().__init__()
        # Patch embedding (3D conv for temporal, but we use single frame)
        self.patch_embed = nn.Conv3d(3, hidden, kernel_size=(2, patch_size, patch_size),
                                      stride=(2, patch_size, patch_size), bias=False)
        # Positional embedding
        self.pos_embed = nn.Embedding(2304, hidden)  # num_position_embeddings

        # ViT blocks
        self.blocks = nn.ModuleList([
            VisionBlock(hidden, heads, mlp_dim) for _ in range(num_blocks)
        ])

        # Merger: spatial 2x2 merge -> project to out_hidden
        self.merger = nn.Module()
        self.merger.norm = nn.LayerNorm(hidden)
        self.merger.linear_fc1 = nn.Linear(hidden * 4, mlp_dim)  # 4096 -> 4096
        self.merger.linear_fc2 = nn.Linear(mlp_dim, out_hidden)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(out_hidden, out_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_hidden, num_classes),
        )

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: [B, C, H, W] image tensor
        Returns:
            logits: [B, num_classes]
        """
        B = pixel_values.shape[0]

        # Patch embed: [B, 3, H, W] -> add temporal dim -> [B, 3, 2, H, W]
        # For single image, duplicate along temporal
        x = pixel_values.unsqueeze(2).expand(-1, -1, 2, -1, -1)
        x = self.patch_embed(x)  # [B, hidden, T', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden]

        # Add positional embeddings
        seq_len = x.shape[1]
        pos_ids = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embed(pos_ids)

        # ViT blocks
        for block in self.blocks:
            x = block(x)

        # Pool and classify
        x = x.mean(dim=1)  # [B, hidden] global average pool
        # Skip merger for classification, go straight to cls head
        # (merger is for projecting to language model space)
        logits = self.cls_head(
            nn.functional.gelu(
                self.merger.linear_fc2(
                    nn.functional.gelu(
                        self.merger.linear_fc1(
                            self.merger.norm(x).repeat(1, 4)  # fake 2x2 merge for single token
                        )
                    )
                )
            )
        )
        return logits


def test_forward():
    """Test that the model can do a forward pass."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = Qwen3VLVisionEncoder(num_blocks=24, num_classes=356)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params/1e6:.1f}M")

    model = model.to(device).to(torch.bfloat16)

    # Test with a dummy image
    img = torch.randn(1, 3, 512, 512, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        logits = model(img)
    print(f"Input: {img.shape}")
    print(f"Output: {logits.shape}")
    print(f"Predicted class: {logits.argmax(dim=-1).item()}")

    gpu_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"GPU memory: {gpu_mb:.0f}MB")
    print("\nFORWARD PASS: OK")


def test_weight_loading():
    """Test loading the extracted Qwen3-VL-2B vision weights."""
    weights_path = Path("/tmp/qwen3vl2b_vision_stripped.pt")
    if not weights_path.exists():
        print("No extracted weights found, skipping load test")
        return

    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    print(f"Loaded {len(state)} weight tensors")

    model = Qwen3VLVisionEncoder(num_blocks=24, num_classes=356)

    # Try loading (may have mismatches due to simplified architecture)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  First 5 missing: {missing[:5]}")
    if unexpected:
        print(f"  First 5 unexpected: {unexpected[:5]}")

    print("\nWEIGHT LOADING: OK" if not missing else "WEIGHT LOADING: PARTIAL")


if __name__ == "__main__":
    test_forward()
    print()
    test_weight_loading()
