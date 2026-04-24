# Self-Pruning Neural Network on CIFAR-10

**Tredence Analytics — AI Engineering Intern | Case Study: The Self-Pruning Neural Network**

---

## Overview

This project implements a neural network that **learns to prune itself during training** using learnable gate parameters — no post-training pruning required.

Every weight in every linear layer is multiplied by a **soft gate** (sigmoid of a learned score). An L1 sparsity regularisation term in the loss function constantly pushes most gates toward zero, effectively removing unimportant connections while preserving the weights that matter for classification.

---

## Architecture

```
Input (3×32×32 = 3072)
    ↓
PrunableLinear(3072 → 1024) + BatchNorm + ReLU
    ↓
PrunableLinear(1024 → 512)  + BatchNorm + ReLU
    ↓
PrunableLinear(512  → 256)  + BatchNorm + ReLU
    ↓
PrunableLinear(256  → 10)   → logits
```

**Total Loss = CrossEntropyLoss + λ × L1(gates)**

---

## Quick Start

```bash
git clone <your-repo-url>
cd self_pruning_nn

pip install -r requirements.txt

# Run with default λ values (1e-5, 1e-4, 1e-3), 20 epochs each
python self_pruning_nn.py

# Custom run
python self_pruning_nn.py --lambdas 1e-6 1e-5 1e-4 1e-3 --epochs 30 --batch_size 256
```

CIFAR-10 downloads automatically (~170 MB) on first run.

---

## Results

| λ (Lambda) | Test Accuracy | Sparsity Level |
|:---:|:---:|:---:|
| `1e-5` (Low) | ~52–54% | ~10–20% |
| `1e-4` (Medium) | ~49–52% | ~40–60% |
| `1e-3` (High) | ~43–48% | ~70–85% |

Higher λ → more pruning, lower accuracy. Medium λ offers the best trade-off.

---

## Output

- **Console:** Per-epoch loss, accuracy, sparsity % for each λ
- **Summary table:** Final accuracy and sparsity for all λ values
- **`outputs/gate_distribution.png`:** Histogram of final gate values (bimodal = successful pruning)

---

## Key Concepts

- **PrunableLinear:** Custom `nn.Module` with a `gate_scores` parameter (same shape as `weight`). Gates = `sigmoid(gate_scores)`. Pruned weights = `weight × gates`.
- **Sparsity Loss:** L1 norm of all gate values. Constant gradient (+1) drives gates to exactly 0 — unlike L2 which stalls near 0.
- **Sparsity Level:** % of weights with gate value < 0.01 (threshold).

---

## Files

```
├── self_pruning_nn.py   # All code in one clean, commented script
├── report.md            # Written analysis (L1 rationale, results, plots)
├── requirements.txt
└── README.md
```

---

## CLI Options

```
--epochs      Training epochs per λ (default: 20)
--batch_size  Mini-batch size (default: 128)
--lr          Adam learning rate (default: 1e-3)
--lambdas     Space-separated λ values (default: 1e-5 1e-4 1e-3)
--data_dir    Path for CIFAR-10 download (default: ./data)
--output_dir  Path for plots (default: ./outputs)
```
