"""
Grokking: Modular Addition Training Script

Train a 1-layer Transformer to learn (a + b) mod p.
Uses wandb for logging and visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from simple_parsing import parse
from dataclasses import asdict

from src.config import GrokkingConfig
from src.data import get_dataloaders
from src.model import ModularAdditionTransformer
from src.utils import get_device, set_seed

import treescope


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


def main():
    # Setup
    config = parse(GrokkingConfig)
    torch.set_float32_matmul_precision("high")
    set_seed(config.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: {config}")

    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        config=asdict(config),
        name=f"p{config.p}_wd{config.weight_decay}_lr{config.lr}",
    )

    # Data
    train_loader, test_loader = get_dataloaders(
        p=config.p,
        train_frac=config.train_frac,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    print(f"Train sample size: {len(train_loader.dataset)}")
    print(f"Test sample size: {len(test_loader.dataset)}")

    # Model
    model = ModularAdditionTransformer(
        p=config.p,
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)

    # uncomment this to create a cool viz of the model using treescope
    # with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
    #     contents = treescope.render_to_html(model)
    # with open("./assets/model.html", "w") as f:
    #     f.write(contents)
    
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    wandb.log({"n_params": n_params})

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
    )

    # Training loop
    pbar = tqdm(range(1, config.epochs + 1), desc="Training")
    for epoch in pbar:
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate and log
        if epoch % config.eval_every == 0 or epoch == 1:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "test/loss": test_loss,
                    "test/accuracy": test_acc,
                }
            )

            pbar.set_postfix(
                {
                    "tr_acc": f"{train_acc:.2%}",
                    "te_acc": f"{test_acc:.2%}",
                }
            )

        elif epoch % config.log_every == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                }
            )

    # Final evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.2%}")

    wandb.finish()


if __name__ == "__main__":
    main()
