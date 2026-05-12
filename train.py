"""
train.py
--------
Unified training + validation loop for AIML331 Assignment 3.

Supports both CNN (Part 1) and ViT (Part 2) models.
Logs loss and accuracy to TensorBoard (mirrors lecturer's SummaryWriter usage).
Saves the best checkpoint by validation accuracy.

Usage (CLI):
    python train.py --model cnn --epochs 20 --img_size 128 --batch_size 32
    python train.py --model vit --epochs 25 --img_size 128 --batch_size 32

Usage (as a function from experiments.py):
    from train import run_training
    results = run_training(model, run_name='cnn_bn_relu_5layers', ...)
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data_loader import get_dataloaders, count_parameters, CLASS_NAMES
from models.cnn import PetCNN
from models.vit import VisionTransformer


# ── Training config defaults ───────────────────────────────────────────────────

CHECKPOINT_DIR = './checkpoints'
RUNS_DIR       = './runs'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)


# ── Core training function ────────────────────────────────────────────────────

def run_training(
    model: nn.Module,
    run_name: str,
    train_loader,
    val_loader,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    device: torch.device = None,
    save_checkpoint: bool = True,
):
    """
    Trains `model` for `num_epochs` epochs, logging to TensorBoard.

    Mirrors the lecturer's training loop structure:
      - CrossEntropyLoss
      - Adam optimiser
      - SummaryWriter for loss (per step) and accuracy (per epoch)
      - Saves best model checkpoint by val accuracy

    Args:
        model:            Any nn.Module (PetCNN or VisionTransformer).
        run_name:         Unique name for TensorBoard run and checkpoint file.
        train_loader:     Training DataLoader.
        val_loader:       Validation DataLoader.
        num_epochs:       Number of training epochs.
        learning_rate:    Adam learning rate.
        device:           torch.device (auto-detected if None).
        save_checkpoint:  Whether to save best model weights to disk.

    Returns:
        dict with keys:
            best_val_acc  (float, 0-100)
            best_epoch    (int)
            total_params  (int)
            history       (dict of lists: train_loss, val_acc per epoch)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # ── Loss and optimiser ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ── LR scheduler: reduce on plateau to help convergence ──
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # ── TensorBoard writer (mirrors lecturer's usage) ──
    writer = SummaryWriter(os.path.join(RUNS_DIR, run_name))

    total_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Run      : {run_name}")
    print(f"Device   : {device}")
    print(f"Params   : {total_params:,}")
    print(f"Epochs   : {num_epochs}  |  LR: {learning_rate}")
    print(f"{'='*60}\n")

    # ── Training state ──
    best_val_acc  = 0.0
    best_epoch    = 0
    history = {'train_loss': [], 'val_acc': []}
    global_step   = 0

    for epoch in range(num_epochs):
        # ────────────── Training phase ──────────────
        model.train()
        running_loss = 0.0
        total_step   = len(train_loader)

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss    = criterion(outputs, labels)

            # Backward pass + optimise
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step  += 1

            # Log loss per step to TensorBoard (matches lecturer's approach)
            writer.add_scalar('Loss/train_step', loss.item(), global_step)

            if (i + 1) % 50 == 0:
                print(f'  Epoch [{epoch+1}/{num_epochs}] '
                      f'Step [{i+1}/{total_step}] '
                      f'Loss: {loss.item():.4f}')

        avg_train_loss = running_loss / total_step
        history['train_loss'].append(avg_train_loss)

        # Log average epoch loss
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # ────────────── Validation phase ──────────────
        val_acc = evaluate(model, val_loader, device)
        history['val_acc'].append(val_acc)

        # Log validation accuracy per epoch
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Step LR scheduler based on val accuracy
        scheduler.step(val_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {avg_train_loss:.4f}  '
              f'Val Acc: {val_acc:.2f}%')

        # ── Save best checkpoint ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch + 1
            if save_checkpoint:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f'{run_name}_best.pth')
                torch.save(model.state_dict(), ckpt_path)
                print(f'  --> New best! Saved checkpoint: {ckpt_path}')

    writer.close()
    print(f'\nTraining complete. Best val acc: {best_val_acc:.2f}% at epoch {best_epoch}')

    return {
        'best_val_acc': round(best_val_acc, 2),
        'best_epoch':   best_epoch,
        'total_params': total_params,
        'history':      history,
    }


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    """
    Evaluates model accuracy on a DataLoader.

    Mirrors the lecturer's test loop (torch.no_grad(), argmax prediction).

    Args:
        model:   Trained nn.Module.
        loader:  DataLoader (val or test).
        device:  Compute device.

    Returns:
        Accuracy as a percentage (0-100).
    """
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs   = model(images)
            _, predicted = torch.max(outputs.data, 1)  # class with highest logit

            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# ── Inference timing helper ───────────────────────────────────────────────────

def measure_inference_time(model: nn.Module, device: torch.device,
                            img_size: int = 128, n_runs: int = 200) -> float:
    """
    Measures average inference time per single image in milliseconds.

    Used for the CNN vs ViT comparison table in the report.

    Args:
        model:    Trained model (must already be on device).
        device:   torch.device.
        img_size: Input image side length.
        n_runs:   Number of forward passes to average over.

    Returns:
        Average milliseconds per image (float).
    """
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    # Warm-up passes (fills CUDA cache, avoids cold-start bias)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    # Timed passes
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)
    elapsed = time.perf_counter() - start

    ms_per_image = (elapsed / n_runs) * 1000
    return round(ms_per_image, 3)


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='AIML331 A3 Training Script')
    parser.add_argument('--model',      type=str,   default='cnn',
                        choices=['cnn', 'vit'], help='Model type to train')
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--img_size',   type=int,   default=128)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--data_path',  type=str,   default='./data')
    parser.add_argument('--num_workers',type=int,   default=2)
    # CNN-specific
    parser.add_argument('--num_layers',    type=int,  default=5)
    parser.add_argument('--use_batchnorm', action='store_true', default=True)
    parser.add_argument('--activation',    type=str,  default='relu')
    parser.add_argument('--use_residual',  action='store_true', default=False)
    # ViT-specific
    parser.add_argument('--embed_dim',  type=int, default=32)
    parser.add_argument('--num_heads',  type=int, default=4)
    parser.add_argument('--vit_layers', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--use_pos_emb',action='store_true', default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── DataLoaders ──
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        img_size=args.img_size,
        batch_size=args.batch_size,
        root_path=args.data_path,
        num_workers=args.num_workers,
    )

    # ── Build model ──
    if args.model == 'cnn':
        model = PetCNN(
            num_classes=4,
            img_size=args.img_size,
            num_layers=args.num_layers,
            use_batchnorm=args.use_batchnorm,
            activation=args.activation,
            use_residual=args.use_residual,
        )
        run_name = (f"cnn_layers{args.num_layers}_"
                    f"{'bn' if args.use_batchnorm else 'nobn'}_"
                    f"{args.activation}_"
                    f"{'res' if args.use_residual else 'std'}")
    else:
        num_patches = (args.img_size // args.patch_size) ** 2
        model = VisionTransformer(
            embed_dim=args.embed_dim,
            hidden_dim=args.embed_dim * 4,
            num_channels=3,
            num_heads=args.num_heads,
            num_layers=args.vit_layers,
            num_classes=4,
            patch_size=args.patch_size,
            num_patches=num_patches,
            use_pos_embedding=args.use_pos_emb,
        )
        run_name = (f"vit_emb{args.embed_dim}_heads{args.num_heads}_"
                    f"layers{args.vit_layers}_patch{args.patch_size}_"
                    f"{'posemb' if args.use_pos_emb else 'noposemb'}")

    # ── Train ──
    results = run_training(
        model=model,
        run_name=run_name,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
    )

    # ── Final test evaluation ──
    # Load best checkpoint before testing
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{run_name}_best.pth')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f'\nLoaded best checkpoint from: {ckpt_path}')

    test_acc = evaluate(model, test_loader, device)
    inf_ms   = measure_inference_time(model, device, img_size=args.img_size)

    print(f'\nFinal Test Accuracy : {test_acc:.2f}%')
    print(f'Inference time      : {inf_ms} ms/image')
    print(f'Best Val Accuracy   : {results["best_val_acc"]}% (epoch {results["best_epoch"]})')
    print(f'Total Parameters    : {results["total_params"]:,}')