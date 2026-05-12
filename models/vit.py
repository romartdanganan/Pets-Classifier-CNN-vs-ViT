# models/vit.py
"""
Vision Transformer (ViT) from scratch for Oxford-IIIT Pets.
Follows lecturer's tutorial exactly: patch embedding, sinusoidal positional
encoding, MultiheadAttention blocks, CLS token, MLP classification head.
"""

import numpy as np
import torch
import torch.nn as nn
import math


# ══════════════════════════════════════════════════════════════════════
# Helper functions (from lecturer's tutorial)
# ══════════════════════════════════════════════════════════════════════

def img_to_patch(x, patch_size):
    """
    Splits image into patch sequence.
    Input:  [B, C, H, W]
    Output: [B, num_patches, C, patch_H, patch_W]
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    return x


def get_sinusoidal_positional_embedding(n_positions, dim):
    """
    Fixed sinusoidal positional embeddings.
    Shape: (1, n_positions, dim)
    (From lecturer's tutorial)
    """
    position = torch.arange(n_positions).unsqueeze(1)  # (n_positions, 1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))

    pe = torch.zeros(n_positions, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0)  # (1, n_positions, dim)


# ══════════════════════════════════════════════════════════════════════
# Transformer components
# ══════════════════════════════════════════════════════════════════════

class AttentionBlock(nn.Module):
    """
    One transformer encoder block (matches lecturer's tutorial):
      LayerNorm → MultiheadAttention (+residual)
      → LayerNorm → FFN with GELU (+residual)
    """

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),  # expansion
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),  # projection back
            nn.Dropout(dropout)
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for 4-class pet classification.

    Args:
        embed_dim:      Patch embedding dimension (default: 32)
        hidden_dim:     FFN hidden dimension (typically 2-4x embed_dim)
        num_channels:   Input channels (3 for RGB)
        num_heads:      Attention heads per block
        num_layers:     Number of transformer blocks
        num_classes:    Output classes (4)
        patch_size:     Patch side length in pixels
        num_patches:    Total patches: (img_size/patch_size)^2
        dropout:        Dropout rate
        use_pos_embedding: Toggle positional embeddings
    """

    def __init__(self, embed_dim=32, hidden_dim=128, num_channels=3,
                 num_heads=4, num_layers=4, num_classes=4,
                 patch_size=8, num_patches=256,
                 dropout=0.0, use_pos_embedding=True):
        super().__init__()

        self.patch_size = patch_size

        # ── Input layer: linear projection of flattened patches ──
        self.input_layer = nn.Linear(num_channels * (patch_size ** 2), embed_dim)

        # ── Transformer blocks (matches lecturer's nn.Sequential) ──
        self.transformer = nn.Sequential(*[
            AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ── Classification head ──
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

        # ── CLS token (learnable) ──
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # ── Positional embedding ──
        if use_pos_embedding:
            self.register_buffer(
                'pos_embedding',
                get_sinusoidal_positional_embedding(1 + num_patches, embed_dim)
            )
        else:
            self.pos_embedding = None

    def forward(self, x):
        # Step 1: Patchify and embed
        x = img_to_patch(x, self.patch_size)  # [B, T, C, Ph, Pw]
        B, T, C, Ph, Pw = x.shape
        x = x.flatten(2, 4)  # [B, T, C*Ph*Pw]
        x = self.input_layer(x)  # [B, T, embed_dim]

        # Step 2: Add CLS token
        cls_token = self.cls_token.repeat(B, 1, 1)  # [B, 1, embed_dim]
        x = torch.cat([cls_token, x], dim=1)  # [B, 1+T, embed_dim]

        # Step 3: Add positional embedding
        if self.pos_embedding is not None:
            x = x + self.pos_embedding[:, :T + 1, :]

        # Step 4: Transformer blocks (expect seq_len first)
        x = self.dropout(x)
        x = x.transpose(0, 1)  # (1+T, B, embed_dim)
        x = self.transformer(x)  # (1+T, B, embed_dim)

        # Step 5: Classification from CLS token
        cls = x[0]  # Get CLS token: (B, embed_dim)
        out = self.mlp_head(cls)  # (B, num_classes)

        return out