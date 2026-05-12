# Pets-Classifier-CNN-vs-ViT


# AIML331 Assignment 3 — CNN vs Vision Transformer for Image Classification

**Course:** AIML331: Computer Vision  
**Dataset:** Oxford-IIIT Pets (4-class wrapper)  
**Framework:** PyTorch  
**Task:** Fine-grained pet breed classification — Long-haired cat, Short-haired cat, Long-haired dog, Short-haired dog

---

## Why CNNs and Transformers are Fundamentally Different

**CNNs (Convolutional Neural Networks)** process images with a sliding kernel — a small filter that
looks at one local neighbourhood at a time. This gives them a strong *inductive bias*: they naturally
assume that nearby pixels are related (locality) and that a pattern is the same wherever it appears
(translation equivariance). This makes them sample-efficient and fast to train on smaller datasets.

**Vision Transformers (ViTs)** take a completely different approach. They split the image into fixed-size
patches and treat each patch as a "token", exactly like a word in a sentence. A self-attention mechanism
then lets *every* patch attend to *every other* patch simultaneously — capturing global context from the
very first layer. This power comes at a cost: ViTs have no built-in spatial bias and typically need
more data and compute to match a well-tuned CNN.

| Property | CNN | Vision Transformer |
|---|---|---|
| Receptive field | Local → grows with depth | Global from layer 1 |
| Inductive bias | Strong (locality, translation) | Weak (learned from data) |
| Parameter efficiency | High on small data | Better at scale |
| Interpretability | Feature maps | Attention maps |
| Typical convergence | Faster | Slower (more epochs) |

---

## Project Structure

```
aiml331-cv-assignment3/
├── dataset_wrapper.py       # Lecturer-provided dataset wrapper
├── models/
│   ├── cnn.py               # Part 1: CNN built from scratch
│   └── vit.py               # Part 2: ViT built from scratch
├── train.py                 # Training + validation loop
├── evaluate.py              # Final test-set evaluation
├── experiments.py           # Ablation experiment runner
├── results/
│   ├── experiment_tracker.py
│   └── experiments.csv      # Auto-logged experiment results
└── images/                  # TensorBoard screenshots for this README
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/aiml331-cv-assignment3.git
cd aiml331-cv-assignment3

# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (select based on your system):
# For CUDA 12.1 (most Windows/Linux with NVIDIA GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# For CPU-only or Mac:
pip install torch torchvision torchaudio
```


---

## Training

```bash
# Train CNN baseline
python train.py --model cnn --epochs 20 --img_size 128

# Train ViT baseline
python train.py --model vit --epochs 25 --img_size 128

# Run all ablation experiments (saves to results/experiments.csv)
python experiments.py
```

---

## Results Summary

### Part 1: CNN

| Experiment | Config | Val Acc | Test Acc |
|---|---|---|---|
| Baseline | BN=True, ReLU, layers=5 | — | — |
| No BatchNorm | BN=False, ReLU, layers=5 | — | — |
| Activation: GELU | BN=True, GELU, layers=5 | — | — |
| Depth ablation | BN=True, ReLU, layers=7 | — | — |
| + Residual | BN=True, ReLU, layers=5, res=True | — | — |

### Part 2: ViT

| Experiment | Config | Val Acc | Test Acc |
|---|---|---|---|
| Baseline | heads=4, layers=4, patch=8 | — | — |
| Heads ablation | heads=6 | — | — |
| Patch size | patch=16 | — | — |
| No pos embedding | pos=False | — | — |

*Results will be filled in after experiments complete.*

---

## TensorBoard Curves

```bash
tensorboard --logdir=runs/
```

<!-- Add screenshots to images/ and link them here as you go -->
<!-- Example: ![CNN Loss Curve](images/cnn_loss_curve.png) -->

---

## CNN vs ViT Comparison

| Metric | CNN | ViT |
|---|---|---|
| Test Accuracy | — | — |
| Parameters | — | — |
| Best Epoch | — | — |
| Inference (ms/img) | — | — |

---

## References

- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016  
- Dosovitskiy et al., *An Image is Worth 16×16 Words*, ICLR 2021  
- Parkhi et al., *Cats and Dogs*, CVPR 2012 (Oxford-IIIT Pets)