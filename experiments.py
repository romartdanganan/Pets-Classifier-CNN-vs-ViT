"""
experiments.py
--------------
Automated ablation study runner for AIML331 Assignment 3.

Runs all required CNN and ViT experiments and logs to results/experiments.csv.

RUNTIME ESTIMATES (CPU, img_size=128, 20 epochs, batch=32):
  CNN experiments  : ~10 runs x ~25 min each = ~4 hours
  ViT experiments  : ~10 runs x ~40 min each = ~7 hours
  Total on CPU     : ~11 hours

RECOMMENDED APPROACH:
  - Use img_size=64 to cut runtime by ~4x (accepted per assignment)
  - Or use a GPU (CUDA)
  - Or run CNN and ViT separately on different days

Usage:
    # Quick smoke test (1 epoch, 64x64 - verifies code runs correctly)
    python experiments.py --img_size 64 --epochs 1 --batch_size 32

    # CNN experiments only (64x64, 15 epochs)
    python experiments.py --img_size 64 --epochs 15 --batch_size 32 --part cnn

    # ViT experiments only
    python experiments.py --img_size 64 --epochs 15 --batch_size 32 --part vit

    # Full run, 128x128 (needs GPU or overnight)
    python experiments.py --img_size 128 --epochs 20 --batch_size 32

NOTE on ViT heads ablation [3, 4, 5, 6] with embed_dim=32:
    PyTorch MultiheadAttention requires embed_dim % num_heads == 0.
    32 is not divisible by 3 or 5. We handle this in VisionTransformer by
    using an internal_dim = nearest multiple of num_heads >= 32, with
    lightweight projections in/out. This is documented in the report.
"""

import argparse
import warnings
import os
import torch

from data_loader import get_dataloaders
from models.cnn import PetCNN
from models.vit import VisionTransformer
from train import run_training
from evaluate import evaluate_accuracy, measure_inference_time, count_parameters
from results.experiment_tracker import ExperimentTracker

warnings.filterwarnings('ignore')

os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./runs', exist_ok=True)


# ── CNN Experiments ────────────────────────────────────────────────────────────

def run_cnn_experiments(args, device, tracker, train_loader, val_loader, test_loader):
    """
    Part 1 — All CNN ablation studies.

    Experiments:
        1. Baseline          : BN=True, ReLU, 5 layers
        2. No BatchNorm      : BN=False, ReLU, 5 layers
        3a. LeakyReLU        : BN=True, LeakyReLU, 5 layers
        3b. GELU             : BN=True, GELU, 5 layers
        4. Depth [3,4,5,6,7] : BN=True, ReLU, varying layers
           (6,7 layers only valid for img_size >= 128)
        5. Residual          : BN=True, ReLU, 5 layers, residual=True
    """
    print("\n" + "=" * 60)
    print("PART 1: CNN ABLATION EXPERIMENTS")
    print("=" * 60)

    def _train_eval_log(model, run_name, exp_name, config_str, extra_fields):
        """Helper: train -> load best ckpt -> evaluate -> log to tracker."""
        results  = run_training(model, run_name, train_loader, val_loader,
                                num_epochs=args.epochs, learning_rate=args.lr, device=device)
        ckpt_path = f'checkpoints/{run_name}_best.pth'
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        test_acc = evaluate_accuracy(model, test_loader, device)
        inf_ms   = measure_inference_time(model, device, args.img_size)
        tracker.log(
            model_type='CNN', experiment=exp_name, config=config_str,
            img_size=args.img_size, batch_size=args.batch_size,
            num_epochs=args.epochs, learning_rate=args.lr,
            best_val_acc=results['best_val_acc'], test_acc=round(test_acc, 2),
            best_epoch=results['best_epoch'],
            total_params=count_parameters(model),
            inference_ms=f"{inf_ms:.3f}",
            **extra_fields
        )
        return test_acc

    # ── 1. Baseline ──
    print("\n--- CNN Exp 1: Baseline (BN, ReLU, 5 layers) ---")
    model = PetCNN(num_classes=4, img_size=args.img_size, num_layers=5,
                   use_batchnorm=True, activation='relu', use_residual=False)
    _train_eval_log(model, 'cnn_baseline', 'Baseline', 'BN=True, ReLU, layers=5',
                    dict(num_layers=5, use_batchnorm=True, activation='relu', use_residual=False))

    # ── 2. No BatchNorm ──
    print("\n--- CNN Exp 2: No BatchNorm ---")
    model = PetCNN(num_classes=4, img_size=args.img_size, num_layers=5,
                   use_batchnorm=False, activation='relu', use_residual=False)
    _train_eval_log(model, 'cnn_no_bn', 'BatchNorm', 'BN=False, ReLU, layers=5',
                    dict(num_layers=5, use_batchnorm=False, activation='relu', use_residual=False))

    # ── 3. Activation functions (LeakyReLU and GELU — ReLU already in baseline) ──
    for act in ['leaky_relu', 'gelu']:
        print(f"\n--- CNN Exp 3: Activation = {act} ---")
        model = PetCNN(num_classes=4, img_size=args.img_size, num_layers=5,
                       use_batchnorm=True, activation=act, use_residual=False)
        _train_eval_log(model, f'cnn_{act}', 'Activation', f'BN=True, {act}, layers=5',
                        dict(num_layers=5, use_batchnorm=True, activation=act, use_residual=False))

    # ── 4. Depth ablation [3, 4, 5, 6, 7] ──
    # 6 and 7 layers need img_size >= 128 (64x64 only supports up to 5 layers safely before
    # feature map becomes 1x1, which makes FC head trivial)
    depths = [3, 4, 5]
    if args.img_size >= 128:
        depths.extend([6, 7])
    else:
        print(f"\n  [Note] img_size={args.img_size}: skipping 6 and 7 layers "
              f"(feature map would be <= 1x1). Use --img_size 128 for full depth ablation.")

    for depth in depths:
        if depth == 5:
            continue  # Already done in baseline
        print(f"\n--- CNN Exp 4: Depth = {depth} layers ---")
        model = PetCNN(num_classes=4, img_size=args.img_size, num_layers=depth,
                       use_batchnorm=True, activation='relu', use_residual=False)
        _train_eval_log(model, f'cnn_layers{depth}', 'NumLayers', f'BN=True, ReLU, layers={depth}',
                        dict(num_layers=depth, use_batchnorm=True, activation='relu', use_residual=False))

    # ── 5. Residual connections ──
    print("\n--- CNN Exp 5: Residual Connections ---")
    model = PetCNN(num_classes=4, img_size=args.img_size, num_layers=5,
                   use_batchnorm=True, activation='relu', use_residual=True)
    _train_eval_log(model, 'cnn_residual', 'Residual', 'BN=True, ReLU, layers=5, residual=True',
                    dict(num_layers=5, use_batchnorm=True, activation='relu', use_residual=True))

    print("\n✅ All CNN experiments complete!")


# ── ViT Experiments ────────────────────────────────────────────────────────────

def run_vit_experiments(args, device, tracker, train_loader, val_loader, test_loader):
    """
    Part 2 — All ViT ablation studies.

    embed_dim=32 is FIXED per assignment requirement.

    Experiments:
        1. Baseline        : embed=32, heads=4, layers=4, patch=8, pos_emb=True
        2. Heads [3,4,5,6] : vary num_heads (uses internal_dim projection for 3,5,6)
        3. Layers [3,5,6]  : vary num_layers (4 done in baseline)
        4. Patch [4,8,16]  : vary patch size (8 done in baseline)
        5. No pos embedding: pos_emb=False

    Note on heads: nn.MultiheadAttention requires embed_dim % num_heads == 0.
    For heads in {3,5,6} with embed_dim=32, VisionTransformer automatically uses
    internal_dim = ceil(32/heads)*heads with in/out projections.
    This is noted in the report.
    """
    print("\n" + "=" * 60)
    print("PART 2: ViT ABLATION EXPERIMENTS (embed_dim=32 fixed)")
    print("=" * 60)

    def _vit_train_eval_log(model, run_name, exp_name, config_str, extra_fields):
        results  = run_training(model, run_name, train_loader, val_loader,
                                num_epochs=args.epochs, learning_rate=args.lr, device=device)
        ckpt_path = f'checkpoints/{run_name}_best.pth'
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        test_acc = evaluate_accuracy(model, test_loader, device)
        inf_ms   = measure_inference_time(model, device, args.img_size)
        tracker.log(
            model_type='ViT', experiment=exp_name, config=config_str,
            img_size=args.img_size, batch_size=args.batch_size,
            num_epochs=args.epochs, learning_rate=args.lr,
            best_val_acc=results['best_val_acc'], test_acc=round(test_acc, 2),
            best_epoch=results['best_epoch'],
            total_params=count_parameters(model),
            inference_ms=f"{inf_ms:.3f}",
            **extra_fields
        )
        return test_acc

    def _make_vit(num_heads, num_layers, patch_size, use_pos_emb=True):
        num_patches = (args.img_size // patch_size) ** 2
        return VisionTransformer(
            embed_dim=32, hidden_dim=128, num_channels=3,
            num_heads=num_heads, num_layers=num_layers, num_classes=4,
            patch_size=patch_size, num_patches=num_patches,
            use_pos_embedding=use_pos_emb,
        )

    # ── 1. Baseline ──
    print("\n--- ViT Exp 1: Baseline (embed=32, heads=4, layers=4, patch=8, pos_emb=True) ---")
    model = _make_vit(num_heads=4, num_layers=4, patch_size=8, use_pos_emb=True)
    _vit_train_eval_log(model, 'vit_baseline', 'Baseline',
                        'embed=32, heads=4, layers=4, patch=8, pos_emb=True',
                        dict(embed_dim=32, num_heads=4, num_layers=4, patch_size=8, use_pos_embedding=True))

    # ── 2. Vary attention heads [3, 4, 5, 6] ──
    # heads=4 done in baseline
    for heads in [3, 5, 6]:
        print(f"\n--- ViT Exp 2: Attention heads = {heads} ---")
        model = _make_vit(num_heads=heads, num_layers=4, patch_size=8, use_pos_emb=True)
        _vit_train_eval_log(model, f'vit_heads{heads}', 'NumHeads',
                            f'embed=32, heads={heads}, layers=4, patch=8',
                            dict(embed_dim=32, num_heads=heads, num_layers=4,
                                 patch_size=8, use_pos_embedding=True))

    # ── 3. Vary layers [3, 4, 5, 6] ──
    # layers=4 done in baseline
    for layers in [3, 5, 6]:
        print(f"\n--- ViT Exp 3: Transformer layers = {layers} ---")
        model = _make_vit(num_heads=4, num_layers=layers, patch_size=8, use_pos_emb=True)
        _vit_train_eval_log(model, f'vit_layers{layers}', 'NumLayers',
                            f'embed=32, heads=4, layers={layers}, patch=8',
                            dict(embed_dim=32, num_heads=4, num_layers=layers,
                                 patch_size=8, use_pos_embedding=True))

    # ── 4. Vary patch size [4, 8, 16] ──
    # patch=8 done in baseline
    # patch=4 at 128x128 -> 1024 patches -> O(n²) attention -> VERY slow on CPU
    # patch=4 at 64x64   -> 256 patches  -> manageable
    # We run patch=4 only at 64x64; if user chose 128, we note it and skip to save time
    for ps in [4, 16]:
        if ps == 4 and args.img_size > 64:
            print(f"\n--- ViT Exp 4: Patch size = 4 (SKIPPED at {args.img_size}x{args.img_size}) ---")
            print(f"  Patch=4 at 128x128 = 1024 patches, O(n²) attention = prohibitively slow on CPU.")
            print(f"  Re-run with --img_size 64 to include this experiment.")
            tracker.log(
                model_type='ViT', experiment='PatchSize',
                config=f'embed=32, heads=4, layers=4, patch=4 [SKIPPED - too slow at {args.img_size}x{args.img_size}]',
                img_size=args.img_size, patch_size=4, num_heads=4, num_layers=4, embed_dim=32,
                notes=f'Skipped: 1024 patches at {args.img_size}x{args.img_size} is O(n²) prohibitive on CPU'
            )
            continue
        print(f"\n--- ViT Exp 4: Patch size = {ps} ---")
        model = _make_vit(num_heads=4, num_layers=4, patch_size=ps, use_pos_emb=True)
        _vit_train_eval_log(model, f'vit_patch{ps}', 'PatchSize',
                            f'embed=32, heads=4, layers=4, patch={ps}',
                            dict(embed_dim=32, num_heads=4, num_layers=4,
                                 patch_size=ps, use_pos_embedding=True))

    # ── 5. No positional embedding ──
    print("\n--- ViT Exp 5: No Positional Embedding ---")
    model = _make_vit(num_heads=4, num_layers=4, patch_size=8, use_pos_emb=False)
    _vit_train_eval_log(model, 'vit_no_posemb', 'PosEmbedding',
                        'embed=32, heads=4, layers=4, patch=8, pos_emb=False',
                        dict(embed_dim=32, num_heads=4, num_layers=4,
                             patch_size=8, use_pos_embedding=False))

    print("\n✅ All ViT experiments complete!")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all ablation experiments')
    parser.add_argument('--img_size',   type=int,   default=128,
                        help='Image size. Use 64 for faster runs (accepted per assignment).')
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--data_path',  type=str,   default='./data')
    parser.add_argument('--num_workers',type=int,   default=2)
    parser.add_argument('--part',       type=str,   default='all',
                        choices=['all', 'cnn', 'vit'],
                        help='Which experiments to run.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device     : {device}")
    print(f"Image size : {args.img_size}x{args.img_size}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"Part       : {args.part}")

    # Warn about expected runtime
    if device.type == 'cpu':
        est_hours = {'cnn': 4, 'vit': 7, 'all': 11}
        h = est_hours.get(args.part, 11)
        scale = (args.img_size / 128) ** 2
        print(f"\n⚠️  Running on CPU with img_size={args.img_size}.")
        print(f"   Estimated runtime: ~{h * scale:.1f} hours.")
        print(f"   Tip: use --img_size 64 to reduce by ~4x.\n")

    # Load data once, reuse for all experiments
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        img_size=args.img_size,
        batch_size=args.batch_size,
        root_path=args.data_path,
        num_workers=args.num_workers,
    )

    tracker = ExperimentTracker()

    if args.part in ('all', 'cnn'):
        run_cnn_experiments(args, device, tracker, train_loader, val_loader, test_loader)

    if args.part in ('all', 'vit'):
        run_vit_experiments(args, device, tracker, train_loader, val_loader, test_loader)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    tracker.print_summary()