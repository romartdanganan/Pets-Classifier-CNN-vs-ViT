# models/cnn.py
"""
CNN model built from scratch for Oxford-IIIT Pets classification.
4 classes: Long-haired cat, Short-haired cat, Long-haired dog, Short-haired dog.

Supports all required ablation experiments:
  - BatchNorm: with / without
  - Activation: ReLU / LeakyReLU / GELU
  - Depth: 3-7 convolutional layers
  - Residual: optional skip connections

Mathes lecturer's PyTorch style: clean nn.Module, clear forward pass.
"""

import torch
import torch.nn as nn


class PetCNN(nn.Module):
    """
    Configurable CNN for 4-class pet breed classification.

    Architecture: Conv blocks → Flatten → FC layers → Output

    Each conv block: Conv2d → (optional BatchNorm) → Activation → MaxPool
    Optional residual: output = Conv(x) + x  (when shapes match)

    Args:
        num_classes:    Output classes (4 for our dataset)
        img_size:       Input image size (128 or 64)
        num_layers:     Number of conv layers (3-7 for experiments)
        use_batchnorm:  Toggle batch normalisation
        activation:     'relu', 'leaky_relu', or 'gelu'
        use_residual:   Add skip connections around conv blocks
    """

    def __init__(self, num_classes=4, img_size=128, num_layers=5,
                 use_batchnorm=True, activation='relu', use_residual=False):
        super().__init__()

        # ── Activation function ──
        if activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}. "
                             f"Choose 'relu', 'leaky_relu', or 'gelu'")

        # ── Build conv layers ──
        # Channel progression: 3 → 16 → 32 → 64 → 128 → 256 → 512
        in_channels = 3  # RGB input
        channels = [16, 32, 64, 128, 256, 512, 512]  # max 7 layers

        self.conv_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()  # for skip connections

        for i in range(num_layers):
            out_channels = channels[i]
            layers = []

            # Conv2d
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1))

            # Optional BatchNorm
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))

            # Activation
            layers.append(act_fn)

            # MaxPool (always included)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            self.conv_layers.append(nn.Sequential(*layers))

            # Projection for residual if channel dimensions differ
            if use_residual and in_channels != out_channels:
                self.residual_projections.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1)
                )
            else:
                self.residual_projections.append(None)

            in_channels = out_channels

        # ── Calculate feature map size after all pooling ──
        # Each MaxPool(2,2) halves spatial dimensions
        feature_size = img_size // (2 ** num_layers)
        n_features = channels[num_layers - 1] * feature_size * feature_size

        # ── Classification head ──
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Forward pass with optional residual connections.
        Residual: output = ConvBlock(x) + projected(x)
        """
        for i, conv_layer in enumerate(self.conv_layers):
            residual = x

            # Main path
            out = conv_layer(x)

            # Residual connection (if enabled and dimensions allow)
            if len(self.residual_projections) > 0:
                proj = self.residual_projections[i]
                if proj is not None:
                    residual = proj(residual)
                # Pool residual to match conv output size
                residual = nn.functional.max_pool2d(residual, kernel_size=2, stride=2)
                out = out + residual

            x = out

        # Classification
        x = self.classifier(x)
        return x