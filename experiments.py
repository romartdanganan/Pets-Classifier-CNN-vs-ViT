# experiments.py
"""
Automated ablation study runner for AIML331 Assignment 3.
Runs all CNN and ViT experiments, logs results to results/experiments.csv.

CNN experiments:
  1. Baseline (BN=True, ReLU, 5 layers)
  2. Without BatchNorm
  3. Different activations (ReLU, LeakyReLU, GELU)
  4. Varying depth (3, 5, 7 layers)
  5. With residual connections

ViT experiments:
  1. Baseline (heads=4, layers=4, patch=8)
  2. Varying attention heads (2, 4, 8)
  3. Different patch sizes (8, 16)
  4. Without positional embeddings

Usage:
    python experiments.py --img_size 128 --epochs 20
    python experiments.py --img_size 64 --epochs 15  # faster for testing
"""

import argparse
import torch
from data_loader import get_dataloaders
from models.cnn import PetCNN
from models.vit import VisionTransformer
from train import run_training
from evaluate import evaluate_accuracy, measure_inference_time, count_parameters
from results.experiment_tracker import ExperimentTracker
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def run_cnn_experiments(args, device, tracker):
    """Part 1: All CNN ablation studies."""
    print("\n" + "=" * 60)
    print("PART 1: CNN ABLATION EXPERIMENTS")
    print("=" * 60)

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        img_size=args.img_size,
        batch_size=args.batch_size
    )

    # ── Experiment 1: Baseline ──
    print("\n--- Exp 1: CNN Baseline (BN, ReLU, 5 layers) ---")
    model = PetCNN(num_classes=4, img_size=args.img_size, num_layers=5,
                   use_batchnorm=True, activation='relu', use_residual=False)
    results = run_training(model, 'cnn_baseline', train_loader, val_loader,
                           num_epochs=args.epochs, learning_rate=args.lr, device=device)

    # Load best checkpoint for test eval
    ckpt = torch.load('checkpoints/cnn_baseline_best.pth', map_location=device)
    model.load_state_dict(ckpt)
    test_acc = evaluate_accuracy(model, test_loader, device)
    inf_ms = measure_inference_time(model, device, args.img_size)

    tracker.log(
        model_type='CNN', experiment='Baseline',
        config='BN=True, ReLU, layers=5',
        num_layers=5, use_batchnorm=True, activation='ReLU', use_residual=False,
        img_size=args.img_size, batch_size=args.batch_size,
        num_epochs=args.epochs, learning_rate=args.lr,
        best_val_acc=results['best_val_acc'], test_acc=test_acc,
        best_epoch=results['best_epoch'], total_params=count_parameters(model),
        inference_ms=f"{inf_ms:.2f}"
    )

    # ── Experiment 2: Without BatchNorm ──
    print("\n--- Exp 2: No BatchNorm ---")
    model = PetCNN(num_classes=4, img_size=args.img_size, num_layers=5,
                   use_batchnorm=False, activation='relu', use_residual=False)
    results = run_training(model, 'cnn_no_bn', train_loader, val_loader,
                           num_epochs=args.epochs, learning_rate=args.lr, device=device)

    ckpt = torch.load('checkpoints/cnn_no_bn_best.pth', map_location=device)
    model.load_state_dict(ckpt)
    test_acc = evaluate_accuracy(model, test_loader, device)
    inf_ms = measure_inference_time(model, device, args.img_size)

    tracker.log(
        model_type='CNN', experiment='No BatchNorm',
        config='BN=False, ReLU, layers=5',
        num_layers=5, use_batchnorm=False, activation='ReLU', use_residual=False,
        img_size=args.img_size, batch_size=args.batch_size,
        num_epochs=args.epochs, learning_rate=args.lr,
        best_val_acc=results['best_val_acc'], test_acc=test_acc,
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

        ckpt = torch.load(f'checkpoints/cnn_{act}_best.pth', map_location=device)
        model.load_state_dict(ckpt)
        test_acc = evaluate_accuracy(model, test_loader, device)
        inf_ms = measure_inference_time(model, device, args.img_size)

        tracker.log(
            model_type='CNN', experiment='Activation',
            config=f'BN=True, {act.upper()}, layers=5',
            num_layers=5, use_batchnorm=True, activation=act, use_residual=False,
            img_size=args.img_size, batch_size=args.batch_size,
            num_epochs=args.epochs, learning_rate=args.lr,
            best_val_acc=results['best_val_acc'], test_acc=test_acc,
            best_epoch=results['best_epoch'], total_params=count_parameters(model),
            inference_ms=f"{inf_ms:.2f}"
        )

    # ── Experiment 4: Varying Depth ──
    depths = [3]
    if args.img_size >= 128:
        depths.append(7)
    else:
        print("\n--- Exp 4: Skipping 7-layer (needs 128×128, currently 64×64) ---")

    for depth in depths:
        print(f"\n--- Exp 4: Depth = {depth} layers ---")
        model = PetCNN(num_classes=4, img_size=args.img_size, num_layers=depth,
                       use_batchnorm=True, activation='relu', use_residual=False)
        results = run_training(model, f'cnn_layers{depth}', train_loader, val_loader,
                               num_epochs=args.epochs, learning_rate=args.lr, device=device)

        ckpt = torch.load(f'checkpoints/cnn_layers{depth}_best.pth', map_location=device)
        model.load_state_dict(ckpt)
        test_acc = evaluate_accuracy(model, test_loader, device)
        inf_ms = measure_inference_time(model, device, args.img_size)

        tracker.log(
            model_type='CNN', experiment='NumLayers',
            config=f'BN=True, ReLU, layers={depth}',
            num_layers=depth, use_batchnorm=True, activation='ReLU', use_residual=False,
            img_size=args.img_size, batch_size=args.batch_size,
            num_epochs=args.epochs, learning_rate=args.lr,
            best_val_acc=results['best_val_acc'], test_acc=test_acc,
            best_epoch=results['best_epoch'], total_params=count_parameters(model),
            inference_ms=f"{inf_ms:.2f}"
        )

    # ── Experiment 5: Residual Connections ──
    print(f"\n--- Exp 5: Residual Connections ---")
    model = PetCNN(num_classes=4, img_size=args.img_size, num_layers=5,
                   use_batchnorm=True, activation='relu', use_residual=True)
    results = run_training(model, f'cnn_residual', train_loader, val_loader,
                           num_epochs=args.epochs, learning_rate=args.lr, device=device)

    ckpt = torch.load(f'checkpoints/cnn_residual_best.pth', map_location=device)
    model.load_state_dict(ckpt)
    test_acc = evaluate_accuracy(model, test_loader, device)
    inf_ms = measure_inference_time(model, device, args.img_size)

    tracker.log(
        model_type='CNN', experiment='Residual',
        config='BN=True, ReLU, layers=5, residual=True',
        num_layers=5, use_batchnorm=True, activation='ReLU', use_residual=True,
        img_size=args.img_size, batch_size=args.batch_size,
        num_epochs=args.epochs, learning_rate=args.lr,
        best_val_acc=results['best_val_acc'], test_acc=test_acc,
        best_epoch=results['best_epoch'], total_params=count_parameters(model),
        inference_ms=f"{inf_ms:.2f}"
    )

    print("\n✅ All CNN experiments complete!")


def run_vit_experiments(args, device, tracker):
    """Part 2: All ViT ablation studies."""
    print("\n" + "=" * 60)
    print("PART 2: ViT ABLATION EXPERIMENTS")
    print("=" * 60)

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        img_size=args.img_size,
        batch_size=args.batch_size
    )
    num_patches = (args.img_size // 8) ** 2

    # ── Experiment 1: ViT Baseline ──
    print("\n--- Exp 1: ViT Baseline (heads=4, layers=4, patch=8) ---")
    model = VisionTransformer(
        embed_dim=128, hidden_dim=512, num_channels=3,
        num_heads=4, num_layers=4, num_classes=4,
        patch_size=8, num_patches=num_patches, use_pos_embedding=True
    )
    results = run_training(model, 'vit_baseline', train_loader, val_loader,
                           num_epochs=args.epochs, learning_rate=args.lr, device=device)

    ckpt = torch.load('checkpoints/vit_baseline_best.pth', map_location=device)
    model.load_state_dict(ckpt)
    test_acc = evaluate_accuracy(model, test_loader, device)
    inf_ms = measure_inference_time(model, device, args.img_size)

    tracker.log(
        model_type='ViT', experiment='Baseline',
        config='heads=4, layers=4, patch=8, pos_emb=True',
        num_layers=4, embed_dim=128, num_heads=4, patch_size=8,
        use_pos_embedding=True,
        img_size=args.img_size, batch_size=args.batch_size,
        num_epochs=args.epochs, learning_rate=args.lr,
        best_val_acc=results['best_val_acc'], test_acc=test_acc,
        best_epoch=results['best_epoch'], total_params=count_parameters(model),
        inference_ms=f"{inf_ms:.2f}"
    )

    # ── Experiment 2: Varying Attention Heads ──
    for heads in [2, 8]:
        print(f"\n--- Exp 2: Attention Heads = {heads} ---")
        model = VisionTransformer(
            embed_dim=128, hidden_dim=512, num_channels=3,
            num_heads=heads, num_layers=4, num_classes=4,
            patch_size=8, num_patches=num_patches, use_pos_embedding=True
        )
        results = run_training(model, f'vit_heads{heads}', train_loader, val_loader,
                               num_epochs=args.epochs, learning_rate=args.lr, device=device)

        ckpt = torch.load(f'checkpoints/vit_heads{heads}_best.pth', map_location=device)
        model.load_state_dict(ckpt)
        test_acc = evaluate_accuracy(model, test_loader, device)
        inf_ms = measure_inference_time(model, device, args.img_size)

        tracker.log(
            model_type='ViT', experiment='NumHeads',
            config=f'heads={heads}, layers=4, patch=8, pos_emb=True',
            num_layers=4, embed_dim=128, num_heads=heads, patch_size=8,
            use_pos_embedding=True,
            img_size=args.img_size, batch_size=args.batch_size,
            num_epochs=args.epochs, learning_rate=args.lr,
            best_val_acc=results['best_val_acc'], test_acc=test_acc,
            best_epoch=results['best_epoch'], total_params=count_parameters(model),
            inference_ms=f"{inf_ms:.2f}"
        )

    # ── Experiment 3: Patch Size ──
    patch_size = 16
    num_patches_16 = (args.img_size // patch_size) ** 2
    print(f"\n--- Exp 3: Patch Size = {patch_size} ---")
    model = VisionTransformer(
        embed_dim=128, hidden_dim=512, num_channels=3,
        num_heads=4, num_layers=4, num_classes=4,
        patch_size=patch_size, num_patches=num_patches_16, use_pos_embedding=True
    )
    results = run_training(model, f'vit_patch{patch_size}', train_loader, val_loader,
                           num_epochs=args.epochs, learning_rate=args.lr, device=device)

    ckpt = torch.load(f'checkpoints/vit_patch{patch_size}_best.pth', map_location=device)
    model.load_state_dict(ckpt)
    test_acc = evaluate_accuracy(model, test_loader, device)
    inf_ms = measure_inference_time(model, device, args.img_size)

    tracker.log(
        model_type='ViT', experiment='PatchSize',
        config=f'heads=4, layers=4, patch={patch_size}, pos_emb=True',
        num_layers=4, embed_dim=128, num_heads=4, patch_size=patch_size,
        use_pos_embedding=True,
        img_size=args.img_size, batch_size=args.batch_size,
        num_epochs=args.epochs, learning_rate=args.lr,
        best_val_acc=results['best_val_acc'], test_acc=test_acc,
        best_epoch=results['best_epoch'], total_params=count_parameters(model),
        inference_ms=f"{inf_ms:.2f}"
    )

    # ── Experiment 4: No Positional Embedding ──
    print(f"\n--- Exp 4: No Positional Embedding ---")
    model = VisionTransformer(
        embed_dim=128, hidden_dim=512, num_channels=3,
        num_heads=4, num_layers=4, num_classes=4,
        patch_size=8, num_patches=num_patches, use_pos_embedding=False
    )
    results = run_training(model, f'vit_no_posemb', train_loader, val_loader,
                           num_epochs=args.epochs, learning_rate=args.lr, device=device)

    ckpt = torch.load(f'checkpoints/vit_no_posemb_best.pth', map_location=device)
    model.load_state_dict(ckpt)
    test_acc = evaluate_accuracy(model, test_loader, device)
    inf_ms = measure_inference_time(model, device, args.img_size)

    tracker.log(
        model_type='ViT', experiment='PosEmb',
        config='heads=4, layers=4, patch=8, pos_emb=False',
        num_layers=4, embed_dim=128, num_heads=4, patch_size=8,
        use_pos_embedding=False,
        img_size=args.img_size, batch_size=args.batch_size,
        num_epochs=args.epochs, learning_rate=args.lr,
        best_val_acc=results['best_val_acc'], test_acc=test_acc,
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

    tracker = ExperimentTracker()

    # Run all experiments
    run_cnn_experiments(args, device, tracker)
    run_vit_experiments(args, device, tracker)

    # Print summary
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    tracker.print_summary()