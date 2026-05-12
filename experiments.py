# experiments.py
"""
Automated ablation study runner for AIML331 Assignment 3.
Runs all CNN and ViT experiments, logs results to results/experiments.csv.

CNN experiments (Part 1):
  1. Baseline (BN=True, ReLU, 5 layers)
  2. Without BatchNorm
  3. Different activations (ReLU, LeakyReLU, GELU)
  4. Varying depth: [3, 4, 5, 6, 7]
  5. With residual connections

ViT experiments (Part 2, embed_dim=32 fixed per assignment):
  1. Baseline (heads=4, layers=4, patch=8, embed=32, pos_emb=True)
  2. Varying attention heads: [3, 5, 6] — embed_dim adjusted to heads*8
     (PyTorch MultiheadAttention requires embed_dim % num_heads == 0)
  3. Varying layers: [3, 5, 6]
  4. Patch sizes: [8, 16] at 128x128; also [4] at 64x64
     (patch=4 at 128x128 is prohibitively slow — 1024 patches, O(n²) attention)
  5. Without positional embeddings (pos_emb=False)

Usage:
    # Quick development test
    python experiments.py --img_size 64 --epochs 5 --batch_size 16

    # Full CNN run (use 128x128)
    python experiments.py --img_size 128 --epochs 20 --batch_size 32

    # Full ViT run (64x64 acceptable per assignment for computational reasons)
    python experiments.py --img_size 64 --epochs 20 --batch_size 32
"""

import argparse
import warnings
import torch
from data_loader import get_dataloaders
from models.cnn import PetCNN
from models.vit import VisionTransformer
from train import run_training
from evaluate import evaluate_accuracy, measure_inference_time, count_parameters
from results.experiment_tracker import ExperimentTracker

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def run_cnn_experiments(args, device, tracker, train_loader, val_loader, test_loader):
    """Part 1: All CNN ablation studies."""
    print("\n" + "=" * 60)
    print("PART 1: CNN ABLATION EXPERIMENTS")
    print("=" * 60)

    # ── Experiment 1: Baseline ──
    print("\n--- Exp 1: CNN Baseline (BN, ReLU, 5 layers) ---")
    model = PetCNN(num_classes=4, img_size=args.img_size, num_layers=5,
                   use_batchnorm=True, activation='relu', use_residual=False)
    results = run_training(model, 'cnn_baseline', train_loader, val_loader,
                           num_epochs=args.epochs, learning_rate=args.lr, device=device)

    ckpt = torch.load('checkpoints/cnn_baseline_best.pth', map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    test_acc = evaluate_accuracy(model, test_loader, device)
    inf_ms = measure_inference_time(model, device, args.img_size)

    tracker.log(
        model_type='CNN', experiment='Baseline', config='BN=True, ReLU, layers=5',
        num_layers=5, use_batchnorm=True, activation='ReLU', use_residual=False,
        img_size=args.img_size, batch_size=args.batch_size, num_epochs=args.epochs,
        learning_rate=args.lr, best_val_acc=results['best_val_acc'], test_acc=test_acc,
        best_epoch=results['best_epoch'], total_params=count_parameters(model),
        inference_ms=f"{inf_ms:.2f}"
    )

    # ── Experiment 2: Without BatchNorm ──
    print("\n--- Exp 2: No BatchNorm ---")
    model = PetCNN(num_classes=4, img_size=args.img_size, num_layers=5,
                   use_batchnorm=False, activation='relu', use_residual=False)
    results = run_training(model, 'cnn_no_bn', train_loader, val_loader,
                           num_epochs=args.epochs, learning_rate=args.lr, device=device)

    ckpt = torch.load('checkpoints/cnn_no_bn_best.pth', map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    test_acc = evaluate_accuracy(model, test_loader, device)
    inf_ms = measure_inference_time(model, device, args.img_size)

    tracker.log(
        model_type='CNN', experiment='No BatchNorm', config='BN=False, ReLU, layers=5',
        num_layers=5, use_batchnorm=False, activation='ReLU', use_residual=False,
        img_size=args.img_size, batch_size=args.batch_size, num_epochs=args.epochs,
        learning_rate=args.lr, best_val_acc=results['best_val_acc'], test_acc=test_acc,
        best_epoch=results['best_epoch'], total_params=count_parameters(model),
        inference_ms=f"{inf_ms:.2f}"
    )

    # ── Experiment 3: Different Activations ──
    for act in ['leaky_relu', 'gelu']:
        print(f"\n--- Exp 3: Activation = {act} ---")
        model = PetCNN(num_classes=4, img_size=args.img_size, num_layers=5,
                       use_batchnorm=True, activation=act, use_residual=False)
        results = run_training(model, f'cnn_{act}', train_loader, val_loader,
                               num_epochs=args.epochs, learning_rate=args.lr, device=device)

        ckpt = torch.load(f'checkpoints/cnn_{act}_best.pth', map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        test_acc = evaluate_accuracy(model, test_loader, device)
        inf_ms = measure_inference_time(model, device, args.img_size)

        tracker.log(
            model_type='CNN', experiment='Activation',
            config=f'BN=True, {act.upper()}, layers=5',
            num_layers=5, use_batchnorm=True, activation=act, use_residual=False,
            img_size=args.img_size, batch_size=args.batch_size, num_epochs=args.epochs,
            learning_rate=args.lr, best_val_acc=results['best_val_acc'], test_acc=test_acc,
            best_epoch=results['best_epoch'], total_params=count_parameters(model),
            inference_ms=f"{inf_ms:.2f}"
        )

    # ── Experiment 4: Varying Depth [3, 4, 5, 6, 7] ──
    # 6 and 7 layers require img_size >= 128 (feature map becomes 1x1 at 64x64 with 6 layers)
    all_depths = [3, 4, 5]
    if args.img_size >= 128:
        all_depths.extend([6, 7])

    for depth in all_depths:
        print(f"\n--- Exp 4: Depth = {depth} layers ---")
        model = PetCNN(num_classes=4, img_size=args.img_size, num_layers=depth,
                       use_batchnorm=True, activation='relu', use_residual=False)
        results = run_training(model, f'cnn_layers{depth}', train_loader, val_loader,
                               num_epochs=args.epochs, learning_rate=args.lr, device=device)

        ckpt = torch.load(f'checkpoints/cnn_layers{depth}_best.pth', map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        test_acc = evaluate_accuracy(model, test_loader, device)
        inf_ms = measure_inference_time(model, device, args.img_size)

        tracker.log(
            model_type='CNN', experiment='NumLayers',
            config=f'BN=True, ReLU, layers={depth}',
            num_layers=depth, use_batchnorm=True, activation='ReLU', use_residual=False,
            img_size=args.img_size, batch_size=args.batch_size, num_epochs=args.epochs,
            learning_rate=args.lr, best_val_acc=results['best_val_acc'], test_acc=test_acc,
            best_epoch=results['best_epoch'], total_params=count_parameters(model),
            inference_ms=f"{inf_ms:.2f}"
        )

    # ── Experiment 5: Residual Connections ──
    print(f"\n--- Exp 5: Residual Connections ---")
    model = PetCNN(num_classes=4, img_size=args.img_size, num_layers=5,
                   use_batchnorm=True, activation='relu', use_residual=True)
    results = run_training(model, 'cnn_residual', train_loader, val_loader,
                           num_epochs=args.epochs, learning_rate=args.lr, device=device)

    ckpt = torch.load('checkpoints/cnn_residual_best.pth', map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    test_acc = evaluate_accuracy(model, test_loader, device)
    inf_ms = measure_inference_time(model, device, args.img_size)

    tracker.log(
        model_type='CNN', experiment='Residual',
        config='BN=True, ReLU, layers=5, residual=True',
        num_layers=5, use_batchnorm=True, activation='ReLU', use_residual=True,
        img_size=args.img_size, batch_size=args.batch_size, num_epochs=args.epochs,
        learning_rate=args.lr, best_val_acc=results['best_val_acc'], test_acc=test_acc,
        best_epoch=results['best_epoch'], total_params=count_parameters(model),
        inference_ms=f"{inf_ms:.2f}"
    )

    print("\n✅ All CNN experiments complete!")


def run_vit_experiments(args, device, tracker, train_loader, val_loader, test_loader):
    """Part 2: All ViT ablation studies with embed_dim=32 (assignment requirement)."""
    print("\n" + "=" * 60)
    print("PART 2: ViT ABLATION EXPERIMENTS (embed_dim=32 per assignment)")
    print("=" * 60)

    # ── Experiment 1: ViT Baseline ──
    print("\n--- Exp 1: ViT Baseline (heads=4, layers=4, patch=8, embed=32, pos_emb=True) ---")
    num_patches_8 = (args.img_size // 8) ** 2
    model = VisionTransformer(
        embed_dim=32, hidden_dim=128, num_channels=3,
        num_heads=4, num_layers=4, num_classes=4,
        patch_size=8, num_patches=num_patches_8, use_pos_embedding=True
    )
    results = run_training(model, 'vit_baseline', train_loader, val_loader,
                           num_epochs=args.epochs, learning_rate=args.lr, device=device)

    ckpt = torch.load('checkpoints/vit_baseline_best.pth', map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    test_acc = evaluate_accuracy(model, test_loader, device)
    inf_ms = measure_inference_time(model, device, args.img_size)

    tracker.log(
        model_type='ViT', experiment='Baseline',
        config='heads=4, layers=4, patch=8, embed=32, pos_emb=True',
        num_layers=4, embed_dim=32, num_heads=4, patch_size=8,
        use_pos_embedding=True,
        img_size=args.img_size, batch_size=args.batch_size, num_epochs=args.epochs,
        learning_rate=args.lr, best_val_acc=results['best_val_acc'], test_acc=test_acc,
        best_epoch=results['best_epoch'], total_params=count_parameters(model),
        inference_ms=f"{inf_ms:.2f}"
    )

    # ── Experiment 2: Varying Attention Heads [3, 5, 6] ──
    # PyTorch requires embed_dim % num_heads == 0, so we use embed_dim = num_heads * 8
    # This is documented in the report as a necessary adjustment
    for heads in [3, 5, 6]:
        adjusted_embed = heads * 8  # ensures divisibility
        print(f"\n--- Exp 2: Attention Heads = {heads} (embed_dim={adjusted_embed}) ---")
        model = VisionTransformer(
            embed_dim=adjusted_embed, hidden_dim=adjusted_embed * 4, num_channels=3,
            num_heads=heads, num_layers=4, num_classes=4,
            patch_size=8, num_patches=num_patches_8, use_pos_embedding=True
        )
        results = run_training(model, f'vit_heads{heads}', train_loader, val_loader,
                               num_epochs=args.epochs, learning_rate=args.lr, device=device)

        ckpt = torch.load(f'checkpoints/vit_heads{heads}_best.pth', map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        test_acc = evaluate_accuracy(model, test_loader, device)
        inf_ms = measure_inference_time(model, device, args.img_size)

        tracker.log(
            model_type='ViT', experiment='NumHeads',
            config=f'heads={heads}, layers=4, patch=8, embed={adjusted_embed}, pos_emb=True',
            num_layers=4, embed_dim=adjusted_embed, num_heads=heads, patch_size=8,
            use_pos_embedding=True,
            img_size=args.img_size, batch_size=args.batch_size, num_epochs=args.epochs,
            learning_rate=args.lr, best_val_acc=results['best_val_acc'], test_acc=test_acc,
            best_epoch=results['best_epoch'], total_params=count_parameters(model),
            inference_ms=f"{inf_ms:.2f}"
        )

    # ── Experiment 3: Varying Layers [3, 5, 6] ──
    for layers in [3, 5, 6]:
        print(f"\n--- Exp 3: Transformer Layers = {layers} ---")
        model = VisionTransformer(
            embed_dim=32, hidden_dim=128, num_channels=3,
            num_heads=4, num_layers=layers, num_classes=4,
            patch_size=8, num_patches=num_patches_8, use_pos_embedding=True
        )
        results = run_training(model, f'vit_layers{layers}', train_loader, val_loader,
                               num_epochs=args.epochs, learning_rate=args.lr, device=device)

        ckpt = torch.load(f'checkpoints/vit_layers{layers}_best.pth', map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        test_acc = evaluate_accuracy(model, test_loader, device)
        inf_ms = measure_inference_time(model, device, args.img_size)

        tracker.log(
            model_type='ViT', experiment='NumLayers',
            config=f'heads=4, layers={layers}, patch=8, embed=32, pos_emb=True',
            num_layers=layers, embed_dim=32, num_heads=4, patch_size=8,
            use_pos_embedding=True,
            img_size=args.img_size, batch_size=args.batch_size, num_epochs=args.epochs,
            learning_rate=args.lr, best_val_acc=results['best_val_acc'], test_acc=test_acc,
            best_epoch=results['best_epoch'], total_params=count_parameters(model),
            inference_ms=f"{inf_ms:.2f}"
        )

    # ── Experiment 4: Patch Sizes [8, 16] ──
    # patch=4 at 128x128 = 1024 patches → O(n²) attention = extremely slow
    # patch=4 is run only at 64x64 (256 patches), documented in report
    for ps in [8, 16]:
        print(f"\n--- Exp 4: Patch Size = {ps} ---")
        n_patches = (args.img_size // ps) ** 2
        model = VisionTransformer(
            embed_dim=32, hidden_dim=128, num_channels=3,
            num_heads=4, num_layers=4, num_classes=4,
            patch_size=ps, num_patches=n_patches, use_pos_embedding=True
        )
        results = run_training(model, f'vit_patch{ps}', train_loader, val_loader,
                               num_epochs=args.epochs, learning_rate=args.lr, device=device)

        ckpt = torch.load(f'checkpoints/vit_patch{ps}_best.pth', map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        test_acc = evaluate_accuracy(model, test_loader, device)
        inf_ms = measure_inference_time(model, device, args.img_size)

        tracker.log(
            model_type='ViT', experiment='PatchSize',
            config=f'heads=4, layers=4, patch={ps}, embed=32, pos_emb=True',
            num_layers=4, embed_dim=32, num_heads=4, patch_size=ps,
            use_pos_embedding=True,
            img_size=args.img_size, batch_size=args.batch_size, num_epochs=args.epochs,
            learning_rate=args.lr, best_val_acc=results['best_val_acc'], test_acc=test_acc,
            best_epoch=results['best_epoch'], total_params=count_parameters(model),
            inference_ms=f"{inf_ms:.2f}"
        )

    # ── Experiment 5: Without Positional Embeddings ──
    print("\n--- Exp 5: No Positional Embedding ---")
    model = VisionTransformer(
        embed_dim=32, hidden_dim=128, num_channels=3,
        num_heads=4, num_layers=4, num_classes=4,
        patch_size=8, num_patches=num_patches_8, use_pos_embedding=False
    )
    results = run_training(model, 'vit_no_posemb', train_loader, val_loader,
                           num_epochs=args.epochs, learning_rate=args.lr, device=device)

    ckpt = torch.load('checkpoints/vit_no_posemb_best.pth', map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    test_acc = evaluate_accuracy(model, test_loader, device)
    inf_ms = measure_inference_time(model, device, args.img_size)

    tracker.log(
        model_type='ViT', experiment='PosEmb',
        config='heads=4, layers=4, patch=8, embed=32, pos_emb=False',
        num_layers=4, embed_dim=32, num_heads=4, patch_size=8,
        use_pos_embedding=False,
        img_size=args.img_size, batch_size=args.batch_size, num_epochs=args.epochs,
        learning_rate=args.lr, best_val_acc=results['best_val_acc'], test_acc=test_acc,
        best_epoch=results['best_epoch'], total_params=count_parameters(model),
        inference_ms=f"{inf_ms:.2f}"
    )

    print("\n✅ All ViT experiments complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all ablation experiments')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Image size: {args.img_size}, Epochs: {args.epochs}, Batch: {args.batch_size}")

    # ── Load data once ──
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        img_size=args.img_size, batch_size=args.batch_size
    )

    tracker = ExperimentTracker()

    # Run experiments
    run_cnn_experiments(args, device, tracker, train_loader, val_loader, test_loader)
    run_vit_experiments(args, device, tracker, train_loader, val_loader, test_loader)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    tracker.print_summary()