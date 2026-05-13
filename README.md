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
Pets-Classifier-CNN-vs-ViT/
├── dataset_wrapper.py      # Lecturer-provided: 37 breeds → 4 classes
├── data_loader.py          # PyTorch DataLoader wrapper
├── train.py                # Training + validation loop with TensorBoard
├── evaluate.py             # Final test-set evaluation
├── experiments.py          # Automated ablation experiment runner
├── models/
│   ├── __init__.py
│   ├── cnn.py              # Part 1: CNN built from scratch
│   └── vit.py              # Part 2: ViT built from scratch
├── results/
│   ├── experiment_tracker.py   # CSV logger for ablation results
│   └── experiments.csv         # Auto-generated experiment log
├── checkpoints/            # Saved model weights (git-ignored)
├── data/                   # Oxford-IIIT Pets dataset (git-ignored)
├── runs/                   # TensorBoard logs (git-ignored)
├── images/                 # TensorBoard screenshots for report
├── requirements.txt
├── .gitignore
└── README.md
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
# Quick smoketest — verifies everything works (~5 min)
python experiments.py --img_size 64 --epochs 1 --batch_size 32

# Full ablation study — all CNN and ViT experiments (~2.5 hrs on RTX 3070)
python experiments.py --img_size 64 --epochs 20 --batch_size 32

# Run only CNN or only ViT experiments
python experiments.py --img_size 64 --epochs 20 --part cnn
python experiments.py --img_size 64 --epochs 20 --part vit

# Evaluate a single trained model
python evaluate.py --model cnn --checkpoint checkpoints/cnn_gelu_best.pth --img_size 64
python evaluate.py --model vit --checkpoint checkpoints/vit_patch4_best.pth --img_size 64
```

---

## Results Summary

All experiments were run at **64×64** resolution, **20 epochs**, batch size **32**, using the **Adam** optimizer (lr=0.001).

### Part 1: CNN Ablation Results

| Experiment       | Config                              | Val Acc  | Test Acc | Params   |
|------------------|-------------------------------------|----------|----------|----------|
| Baseline         | BN=True, ReLU, layers=5             | 68.07%   | 67.18%   | 672K     |
| No BatchNorm     | BN=False, ReLU, layers=5            | 58.12%   | 55.59%   | 672K     |
| LeakyReLU        | BN=True, LeakyReLU, layers=5        | 66.95%   | 63.83%   | 672K     |
| GELU             | BN=True, GELU, layers=5             | 66.39%   | **68.58%** | 672K   |
| Depth=3          | BN=True, ReLU, layers=3             | 63.73%   | 65.08%   | 1.09M    |
| Depth=4          | BN=True, ReLU, layers=4             | 67.09%   | **68.58%** | **639K** |
| Residual         | BN=True, ReLU, layers=5, res=True   | 63.31%   | 64.11%   | 716K     |

**Key findings:** BatchNorm gives a massive ~12% boost. GELU and 4-layer CNN both achieve the best test accuracy (**68.58%**). The 4-layer model is the most parameter-efficient.

### Part 2: ViT Ablation Results

| Experiment      | Config                            | Val Acc  | Test Acc | Params |
|-----------------|-----------------------------------|----------|----------|--------|
| Baseline        | heads=4, layers=4, patch=8, pos_emb=True | 47.20%   | **47.07%** | 57K    |
| Heads=3         | heads=3                           | 38.80%   | 42.46%   | 63K    |
| Heads=5         | heads=5                           | 41.46%   | 42.32%   | 69K    |
| Heads=6         | heads=6                           | 43.98%   | 41.76%   | 73K    |
| Layers=3        | layers=3                          | 45.38%   | 45.25%   | 45K    |
| Layers=5        | layers=5                          | 43.42%   | 45.53%   | 70K    |
| Layers=6        | layers=6                          | 47.34%   | 47.07%   | 83K    |
| Patch=4         | patch=4                           | **47.62%** | 46.65%   | 53K    |
| Patch=16        | patch=16                          | 42.72%   | 42.18%   | 76K    |
| No Pos Emb      | pos_emb=False                     | 40.62%   | 43.58%   | 57K    |

**Key findings:** Smaller patches (patch=4) give the best validation accuracy. Positional embeddings are important (~4% gain). Best test accuracy is **47.07%**.

---

## TensorBoard Curves

```bash
tensorboard --logdir=runs/
```

<!-- Add screenshots to images/ and link them here as you go -->
<!-- Example: ![CNN Loss Curve](images/cnn_loss_curve.png) -->

---

## CNN vs ViT Comparison

| Metric           | Best CNN (GELU / Depth=4)       | Best ViT (Baseline) |
|------------------|---------------------------------|---------------------|
| Test Accuracy    | **68.58%**                      | 47.07%              |
| Parameters       | 639K – 672K                     | **57K**             |
| Best Epoch       | 20                              | 19                  |
| Convergence      | Faster & more stable            | Slower              |

**Key takeaway:** The CNN significantly outperforms the ViT by **~21.5 percentage points** on this relatively small dataset. This aligns with expectations, as Vision Transformers typically require much larger datasets to shine. However, the ViT achieves decent performance with **~12× fewer parameters**, showcasing its parameter efficiency.

---

## References

- Parkhi et al., *Cats and Dogs*, CVPR 2012 (Oxford-IIIT Pets)
- Vaswani et al., Attention Is All You Need, NeurIPS 2017
- He et al., Deep Residual Learning for Image Recognition, CVPR 2016