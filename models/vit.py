"""
models/vit.py
-------------
Vision Transformer (ViT) from scratch for AIML331 Assignment 3, Part 2.

Follows the lecturer's tutorial style:
  - img_to_patch helper function
  - Sinusoidal positional embeddings (fixed, not learned)
  - AttentionBlock with pre-norm (LayerNorm before attention)
  - CLS token for classification
  - MLP head on CLS token output

FIX vs previous version:
  - embed_dim divisibility constraint: PyTorch nn.MultiheadAttention requires
    embed_dim % num_heads == 0. The assignment fixes embed_dim=32 but also
    asks to vary num_heads over [3, 4, 5, 6]. Since 32 % 3 != 0 and 32 % 5 != 0,
    we handle this via a lightweight input projection:
      - The patch linear always projects to an internal dim that IS divisible by num_heads
      - We pick internal_dim = smallest multiple of num_heads >= embed_dim
      - This is documented in the report as a necessary engineering adjustment
    For num_heads in [4, 8]: internal_dim = embed_dim = 32 (no overhead)
    For num_heads = 3:       internal_dim = 33
    For num_heads = 5:       internal_dim = 35
    For num_heads = 6:       internal_dim = 36

    Alternatively, for the heads ablation, we can just note in the report
    that valid heads for embed=32 are those where 32 % num_heads == 0,
    i.e., [1, 2, 4, 8, 16, 32]. We test [4] as baseline and add [8] —
    and for [3, 5, 6] we use the adjusted internal_dim approach above.
"""

import numpy as np
import torch
import torch.nn as nn
import math


# ── Helpers (directly from lecturer's tutorial) ───────────────────────────────

def img_to_patch(x, patch_size):
    """
    Splits image into patch sequence.
    Input:  [B, C, H, W]
    Output: [B, num_patches, C, patch_H, patch_W]
    where num_patches = (H/patch_size) * (W/patch_size)
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)   # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)                # [B, H'*W', C, p_H, p_W]
    return x


def get_sinusoidal_positional_embedding(n_positions, dim):
    """
    Fixed sinusoidal positional embeddings (from lecturer's tutorial).
    Returns shape (1, n_positions, dim).
    """
    position = torch.arange(n_positions).unsqueeze(1)           # (n, 1)
    div_term = torch.exp(
        torch.arange(0, dim, 2) * (-np.log(10000.0) / dim)
    )                                                            # (dim/2,)
    pe = torch.zeros(n_positions, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)                                       # (1, n, dim)


# ── Transformer block (matches lecturer's AttentionBlock exactly) ─────────────

class AttentionBlock(nn.Module):
    """
    One transformer encoder block:
        x = x + Attention(LayerNorm(x))     <- self-attention with residual
        x = x + FFN(LayerNorm(x))           <- feed-forward with residual

    This is the pre-norm variant (LayerNorm before sub-layer),
    which is more stable than post-norm for training from scratch.
    """

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Args:
            embed_dim:  Token dimension (must be divisible by num_heads).
            hidden_dim: FFN hidden dimension (typically 2-4x embed_dim).
            num_heads:  Number of attention heads.
            dropout:    Dropout rate.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm attention + residual
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        # Pre-norm FFN + residual
        x = x + self.linear(self.layer_norm_2(x))
        return x


# ── Vision Transformer ────────────────────────────────────────────────────────

class VisionTransformer(nn.Module):
    """
    ViT for 4-class pet classification.

    Assignment constraint: embed_dim=32 (fixed for all experiments).
    Heads ablation challenge: 32 is not divisible by 3 or 5.
    Solution: internal_dim = ceil(embed_dim / num_heads) * num_heads
    An extra Linear projection maps embed_dim -> internal_dim if needed.

    Args:
        embed_dim:         Patch projection dimension (32, fixed per assignment).
        hidden_dim:        FFN hidden size in transformer blocks.
        num_channels:      Input image channels (3 for RGB).
        num_heads:         Attention heads per block.
        num_layers:        Number of transformer encoder blocks.
        num_classes:       Output classes (4).
        patch_size:        Patch side length in pixels.
        num_patches:       Total patches = (img_size / patch_size)^2.
        dropout:           Dropout rate.
        use_pos_embedding: Toggle sinusoidal positional embedding.
    """

    def __init__(
        self,
        embed_dim=32,
        hidden_dim=128,
        num_channels=3,
        num_heads=4,
        num_layers=4,
        num_classes=4,
        patch_size=8,
        num_patches=256,
        dropout=0.0,
        use_pos_embedding=True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim  = embed_dim

        # ── Resolve internal dim (handles embed_dim % num_heads != 0) ──
        # For the heads ablation, we keep embed_dim=32 as the "public" dimension
        # but use internal_dim = nearest multiple of num_heads >= embed_dim.
        if embed_dim % num_heads == 0:
            self.internal_dim = embed_dim
        else:
            # Round up to nearest multiple of num_heads
            self.internal_dim = math.ceil(embed_dim / num_heads) * num_heads
            print(f"  [ViT] embed_dim={embed_dim} not divisible by num_heads={num_heads}. "
                  f"Using internal_dim={self.internal_dim} with projection.")

        # ── Patch linear: projects flattened patch -> embed_dim ──
        self.input_layer = nn.Linear(num_channels * (patch_size ** 2), embed_dim)

        # ── Optional projection to internal_dim if different from embed_dim ──
        if self.internal_dim != embed_dim:
            self.dim_proj = nn.Linear(embed_dim, self.internal_dim, bias=False)
            self.dim_proj_back = nn.Linear(self.internal_dim, embed_dim, bias=False)
        else:
            self.dim_proj = nn.Identity()
            self.dim_proj_back = nn.Identity()

        # ── Transformer blocks ──
        internal_hidden = self.internal_dim * 4
        self.transformer = nn.Sequential(*[
            AttentionBlock(self.internal_dim, internal_hidden, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ── Classification head ──
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        self.dropout = nn.Dropout(dropout)

        # ── CLS token ──
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # ── Positional embedding (sinusoidal, fixed) ──
        if use_pos_embedding:
            self.register_buffer(
                'pos_embedding',
                get_sinusoidal_positional_embedding(1 + num_patches, embed_dim)
            )
        else:
            self.pos_embedding = None

        print(f"  [ViT] patch={patch_size} | embed={embed_dim} | internal={self.internal_dim} | "
              f"heads={num_heads} | layers={num_layers} | pos_emb={use_pos_embedding} | "
              f"patches={num_patches}")

    def forward(self, x):
        # Step 1: Patchify
        x = img_to_patch(x, self.patch_size)   # [B, T, C, Ph, Pw]
        B, T, C, Ph, Pw = x.shape
        x = x.flatten(2, 4)                     # [B, T, C*Ph*Pw]

        # Step 2: Linear patch embedding -> embed_dim
        x = self.input_layer(x)                 # [B, T, embed_dim]

        # Step 3: Prepend CLS token
        cls = self.cls_token.repeat(B, 1, 1)    # [B, 1, embed_dim]
        x   = torch.cat([cls, x], dim=1)        # [B, 1+T, embed_dim]

        # Step 4: Add positional embedding
        if self.pos_embedding is not None:
            x = x + self.pos_embedding[:, :T + 1, :]

        # Step 5: Project to internal_dim if needed
        x = self.dropout(x)
        x = self.dim_proj(x)                    # [B, 1+T, internal_dim]

        # Step 6: Transformer (expects seq-first)
        x = x.transpose(0, 1)                   # [1+T, B, internal_dim]
        x = self.transformer(x)                 # [1+T, B, internal_dim]
        x = x.transpose(0, 1)                   # [B, 1+T, internal_dim]

        # Step 7: Project back to embed_dim
        x = self.dim_proj_back(x)               # [B, 1+T, embed_dim]

        # Step 8: CLS token -> classification head
        cls_out = x[:, 0]                       # [B, embed_dim]
        out     = self.mlp_head(cls_out)        # [B, num_classes]
        return out


if __name__ == '__main__':
    dummy = torch.randn(4, 3, 128, 128)

    configs = [
        dict(embed_dim=32, num_heads=4, num_layers=4, patch_size=8,  num_patches=256),
        dict(embed_dim=32, num_heads=3, num_layers=4, patch_size=8,  num_patches=256),
        dict(embed_dim=32, num_heads=6, num_layers=4, patch_size=8,  num_patches=256),
        dict(embed_dim=32, num_heads=4, num_layers=4, patch_size=4,  num_patches=1024),
        dict(embed_dim=32, num_heads=4, num_layers=4, patch_size=16, num_patches=64),
    ]

    for cfg in configs:
        m   = VisionTransformer(**cfg)
        out = m(dummy)
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  Output: {out.shape} | Params: {params:,}\n")