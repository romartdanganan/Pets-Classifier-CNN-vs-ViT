"""
models/cnn.py
-------------
CNN built from scratch for AIML331 Assignment 3, Part 1.

Supports all ablation experiments:
  - BatchNorm on/off
  - Activation: 'relu', 'leaky_relu', 'gelu'
  - Depth: num_layers in [3, 4, 5, 6, 7]
  - Residual: output = Conv(BN(Act(x))) + x  (matches assignment diagram exactly)

FIX vs previous version:
  - Residual architecture now correctly matches the assignment diagram.
    The diagram shows: skip added BEFORE MaxPool, so spatial dims always match.
  - MaxPool applied AFTER residual addition.
  - Projection conv on skip handles channel mismatch.
"""

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """Returns activation module by name string."""
    mapping = {
        'relu':       nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
        'gelu':       nn.GELU(),
    }
    if name not in mapping:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(mapping.keys())}")
    return mapping[name]


class ConvBlock(nn.Module):
    """
    Standard conv block (no residual):
        Conv2d -> [BatchNorm2d] -> Activation -> MaxPool2d
    """
    def __init__(self, in_ch, out_ch, use_batchnorm=True, activation='relu'):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(get_activation(activation))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    """
    Residual conv block matching assignment diagram exactly:

        Main path:  Conv -> [BN] -> Act
        Skip path:  1x1 projection (if channels differ), else identity
        Addition:   main_out + skip           <- spatial dims match here
        Then:       MaxPool2d(2,2)            <- pool AFTER addition

    So: output = MaxPool( Act(BN(Conv(x))) + skip(x) )

    The key constraint: addition before pool means both tensors are [B, out_ch, H, W].
    """
    def __init__(self, in_ch, out_ch, use_batchnorm=True, activation='relu'):
        super().__init__()

        # Main path (no pool)
        main = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_batchnorm)]
        if use_batchnorm:
            main.append(nn.BatchNorm2d(out_ch))
        main.append(get_activation(activation))
        self.main_path = nn.Sequential(*main)

        # Skip projection: match channels only (spatial dims unchanged here)
        self.skip_proj = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

        # Pool after addition
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        main_out = self.main_path(x)   # [B, out_ch, H, W]
        skip     = self.skip_proj(x)   # [B, out_ch, H, W]  <- same spatial size
        out      = main_out + skip     # residual addition
        return self.pool(out)          # [B, out_ch, H/2, W/2]


class PetCNN(nn.Module):
    """
    Configurable CNN for 4-class Oxford-IIIT Pets classification.

    Channel progression:
        Layer:  1    2    3    4    5    6    7
        Out ch: 16   32   64  128  256  512  512

    Args:
        num_classes:    4 for this assignment.
        img_size:       128 (or 64).
        num_layers:     Conv blocks in range [3, 7].
        use_batchnorm:  Toggle BatchNorm for ablation.
        activation:     'relu', 'leaky_relu', or 'gelu'.
        use_residual:   Use ResidualConvBlock instead of ConvBlock.
    """

    CHANNELS = [3, 16, 32, 64, 128, 256, 512, 512]

    def __init__(self, num_classes=4, img_size=128, num_layers=5,
                 use_batchnorm=True, activation='relu', use_residual=False):
        super().__init__()

        # Validate depth
        max_layers = 0
        s = img_size
        while s >= 2:
            s //= 2
            max_layers += 1
        if num_layers > max_layers:
            raise ValueError(
                f"Cannot use {num_layers} layers with {img_size}x{img_size}. "
                f"Max is {max_layers}. Use img_size=128 for 7-layer experiments."
            )

        BlockClass = ResidualConvBlock if use_residual else ConvBlock

        blocks = []
        for i in range(num_layers):
            blocks.append(BlockClass(
                in_ch=self.CHANNELS[i],
                out_ch=self.CHANNELS[i + 1],
                use_batchnorm=use_batchnorm,
                activation=activation,
            ))
        self.features = nn.Sequential(*blocks)

        # FC input: channels * spatial^2
        feature_map_size = img_size // (2 ** num_layers)
        fc_in = self.CHANNELS[num_layers] * feature_map_size * feature_map_size

        print(f"  [PetCNN] layers={num_layers} | bn={use_batchnorm} | act={activation} | "
              f"res={use_residual} | feature_map={feature_map_size}x{feature_map_size} | fc_in={fc_in}")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    dummy = torch.randn(4, 3, 128, 128)
    for use_res in [False, True]:
        m = PetCNN(num_layers=5, use_batchnorm=True, activation='relu', use_residual=use_res)
        out = m(dummy)
        print(f"  res={use_res} -> output {out.shape}")