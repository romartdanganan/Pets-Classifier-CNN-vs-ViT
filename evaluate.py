# evaluate.py
"""
Final test-set evaluation for trained models.
Loads the best checkpoint and reports test accuracy + inference time.

Usage:
    python evaluate.py --model cnn --checkpoint checkpoints/cnn_best.pth
    python evaluate.py --model vit --checkpoint checkpoints/vit_best.pth
"""

import argparse
import time
import torch
from data_loader import get_dataloaders
from models.cnn import PetCNN
from models.vit import VisionTransformer


def evaluate_accuracy(model, loader, device):
    """Calculate classification accuracy (0-100)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def measure_inference_time(model, device, img_size=128, n_runs=200):
    """Average inference time per image in milliseconds."""
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    # Warm-up (fills CUDA cache, avoids cold-start bias)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)
    elapsed = time.perf_counter() - start

    return round((elapsed / n_runs) * 1000, 3)


def count_parameters(model):
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Final test evaluation')
    parser.add_argument('--model', type=str, required=True,
                        choices=['cnn', 'vit'])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pth checkpoint')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load test data
    _, _, test_loader, class_names = get_dataloaders(
        img_size=args.img_size,
        batch_size=args.batch_size
    )

    # Build model
    if args.model == 'cnn':
        model = PetCNN(num_classes=4, img_size=args.img_size)
    else:
        num_patches = (args.img_size // 8) ** 2
        model = VisionTransformer(
            embed_dim=128, hidden_dim=512, num_channels=3,
            num_heads=4, num_layers=4, num_classes=4,
            patch_size=8, num_patches=num_patches
        )

    # Load trained weights
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate
    test_acc = evaluate_accuracy(model, test_loader, device)
    inf_ms = measure_inference_time(model, device, args.img_size)
    params = count_parameters(model)

    print(f"\n{'='*50}")
    print(f"  Model:            {args.model.upper()}")
    print(f"  Test Accuracy:    {test_acc:.2f}%")
    print(f"  Parameters:       {params:,}")
    print(f"  Inference time:   {inf_ms} ms/image")
    print(f"{'='*50}")

    # Check minimum accuracy requirement
    minimum = 45 if args.model == 'cnn' else 35
    if test_acc >= minimum:
        print(f"  ✅ Meets minimum accuracy ({minimum}%)")
    else:
        print(f"  ❌ Below minimum accuracy ({minimum}%) — needs improvement")