"""
Self-Pruning Neural Network on CIFAR-10

Case Study – AI Engineer | Tredence Analytics

This script implements a neural network that learns to prune itself *during*
training using learnable gate parameters. Each weight in every linear layer
is multiplied by a soft gate (sigmoid of a learned score). An L1 sparsity
regularization term in the loss pushes most gates toward zero, effectively
removing unimportant connections without any post-training pruning step.

Author: Arshita
"""

import os
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless rendering — safe for servers
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# Part 1 – PrunableLinear Layer

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a learnable gate for every
    weight element.

    During the forward pass:
        gates         = sigmoid(gate_scores)          # values in (0, 1)
        pruned_weights = weight * gates               # element-wise mask
        output         = x @ pruned_weights.T + bias

    Because gate_scores is registered as a nn.Parameter, it receives gradients
    and is updated by the optimizer just like the ordinary weights. When a gate
    value collapses to ≈0, the corresponding weight has no effect on the
    output — the connection is "pruned."
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight and bias parameters (same as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features)) if bias else None

        #  Gate scores: same shape as weight 
        # Initialise near 0.5 (sigmoid(0) = 0.5) so training starts with all
        # connections roughly half-open, giving the optimiser room to push
        # scores up or down.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Initialise weight and bias with the same scheme used by nn.Linear
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Convert raw scores → gates in (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: Element-wise mask applied to weights
        pruned_weights = self.weight * gates

        # Step 3: Standard linear projection with pruned weights
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached from the computation graph)."""
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# Self-Pruning Network Definition

class SelfPruningNet(nn.Module):
    """
    A feed-forward network for CIFAR-10 image classification built entirely
    from PrunableLinear layers.

    Architecture:
        Input (3×32×32 = 3072)  →  FC-1024  →  BN  →  ReLU
                                →  FC-512   →  BN  →  ReLU
                                →  FC-256   →  BN  →  ReLU
                                →  FC-10    (logits)
    """

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3 * 32 * 32, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)          # flatten spatial dims
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

    def prunable_layers(self):
        """Iterate over all PrunableLinear layers in the network."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module


# Part 2 – Sparsity Regularisation Loss

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    Compute the L1 norm of all gate values across every PrunableLinear layer.

    Why L1?
    -------
    The L1 norm sums absolute values.  Because gates live in (0,1) — always
    positive — this equals the plain sum of all gate values.  The gradient of
    |g| with respect to g is +1 for g > 0, which means the loss constantly
    pulls every gate toward zero.  L2 would instead pull gates toward *near*
    zero but rarely exactly zero; L1 is the standard choice for inducing exact
    sparsity (cf. Lasso regression, group sparsity in deep learning).

    Total Loss = CrossEntropy + λ × SparsityLoss
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)   # keep in computation graph
        total = total + gates.abs().sum()           # L1 norm (= plain sum here)
    return total


# Part 3 – Data Loading

def get_dataloaders(data_dir: str = "./data", batch_size: int = 128):
    """
    Download CIFAR-10 and return train / test DataLoaders.
    Applies standard normalisation (ImageNet-style mean/std works well for
    CIFAR-10 too) plus random crop + horizontal flip for training augmentation.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# Training & Evaluation helpers

def train_one_epoch(model, loader, optimizer, device, lam: float):
    """Run one full epoch and return (avg_total_loss, avg_ce_loss, avg_sp_loss)."""
    model.train()
    total_loss_sum = ce_loss_sum = sp_loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits    = model(images)
        ce        = criterion(logits, labels)
        sp        = sparsity_loss(model)
        loss      = ce + lam * sp

        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item()
        ce_loss_sum    += ce.item()
        sp_loss_sum    += sp.item()

    n = len(loader)
    return total_loss_sum / n, ce_loss_sum / n, sp_loss_sum / n


@torch.no_grad()
def evaluate(model, loader, device):
    """Return classification accuracy on the given DataLoader."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


@torch.no_grad()
def compute_sparsity(model, threshold: float = 1e-2) -> float:
    """
    Sparsity Level = fraction of weights whose gate value < threshold.
    A gate < 0.01 contributes less than 1% of the original weight's magnitude
    and is considered effectively pruned.
    """
    pruned = total = 0
    for layer in model.prunable_layers():
        gates = layer.get_gates()
        pruned += (gates < threshold).sum().item()
        total  += gates.numel()
    return pruned / total if total > 0 else 0.0


@torch.no_grad()
def collect_all_gates(model) -> np.ndarray:
    """Return a flat numpy array of all gate values across the network."""
    all_gates = []
    for layer in model.prunable_layers():
        all_gates.append(layer.get_gates().cpu().numpy().ravel())
    return np.concatenate(all_gates)


# Run a single experiment for one λ value

def run_experiment(lam: float, train_loader, test_loader, device,
                   epochs: int = 20, lr: float = 1e-3) -> dict:
    """
    Train a fresh SelfPruningNet with the given λ and return a results dict.
    """
    print(f"\n{'='*60}")
    print(f"  λ = {lam}   |   epochs = {epochs}")
    print(f"{'='*60}")

    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Cosine LR decay helps the gates settle cleanly by the end of training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        tl, ce, sp = train_one_epoch(model, train_loader, optimizer, device, lam)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, test_loader, device)
            sparsity = compute_sparsity(model)
            print(f"  Epoch {epoch:3d} | Total Loss: {tl:.4f} | "
                  f"CE: {ce:.4f} | Sp: {sp:.4f} | "
                  f"Acc: {acc*100:.2f}% | Sparsity: {sparsity*100:.1f}%")

    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = compute_sparsity(model)
    all_gates      = collect_all_gates(model)

    print(f"\n  ✔  Final Test Accuracy : {final_acc*100:.2f}%")
    print(f"  ✔  Sparsity Level      : {final_sparsity*100:.1f}%")

    return {
        "lambda":   lam,
        "accuracy": final_acc,
        "sparsity": final_sparsity,
        "gates":    all_gates,
        "model":    model,
    }

# Plotting

def plot_gate_distribution(results: list, save_path: str = "gate_distribution.png"):
    """
    Histogram of final gate values for each λ.  A successful self-pruning run
    shows a large spike at 0 (pruned weights) and a separate cluster near 1
    (kept weights).
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        gates = res["gates"]
        ax.hist(gates, bins=80, color="steelblue", edgecolor="none", alpha=0.85)
        ax.set_title(
            f"λ = {res['lambda']}\n"
            f"Acc = {res['accuracy']*100:.1f}%  |  "
            f"Sparsity = {res['sparsity']*100:.1f}%",
            fontsize=11
        )
        ax.set_xlabel("Gate Value", fontsize=10)
        ax.set_ylabel("Count",      fontsize=10)
        ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.2,
                   label="Prune threshold (0.01)")
        ax.legend(fontsize=8)

    fig.suptitle("Distribution of Gate Values After Training\n"
                 "(Spike at 0 = pruned connections, cluster near 1 = kept connections)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Gate distribution plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Self-Pruning Neural Network on CIFAR-10")
    parser.add_argument("--epochs",     type=int,   default=20,
                        help="Training epochs per λ (default: 20)")
    parser.add_argument("--batch_size", type=int,   default=128,
                        help="Mini-batch size (default: 128)")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--lambdas",    type=float, nargs="+",
                        default=[1e-5, 1e-4, 1e-3],
                        help="Sparsity λ values to compare "
                             "(default: 1e-5 1e-4 1e-3)")
    parser.add_argument("--data_dir",   type=str,   default="./data",
                        help="Directory for CIFAR-10 download")
    parser.add_argument("--output_dir", type=str,   default="./outputs",
                        help="Directory for plots and results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥  Device: {device}")

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)

    results = []
    for lam in args.lambdas:
        res = run_experiment(
            lam=lam,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
        )
        results.append(res)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"{'Lambda':<12} {'Test Accuracy':>15} {'Sparsity Level':>16}")
    print("-"*55)
    for r in results:
        print(f"{r['lambda']:<12} {r['accuracy']*100:>14.2f}% {r['sparsity']*100:>15.1f}%")
    print("="*55)

    # ── Gate distribution plot ───────────────────────────────────────────────
    best = max(results, key=lambda r: r["accuracy"])
    plot_path = os.path.join(args.output_dir, "gate_distribution.png")
    plot_gate_distribution(results, save_path=plot_path)

    print(f"\n✅  Best λ = {best['lambda']} → "
          f"Accuracy = {best['accuracy']*100:.2f}%, "
          f"Sparsity = {best['sparsity']*100:.1f}%")
    print("\nDone! Check the outputs/ folder for the gate distribution plot.")


if __name__ == "__main__":
    main()
