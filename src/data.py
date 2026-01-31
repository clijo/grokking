"""Data generation for modular addition task."""

import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_modular_addition_data(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate all p^2 pairs for modular addition.

    Args:
        p: Prime modulus

    Returns:
        inputs: Tensor of shape (p^2, 2) with all (a, b) pairs
        labels: Tensor of shape (p^2,) with (a + b) mod p
    """
    a = torch.arange(p)
    b = torch.arange(p)

    # Create all p^2 pairs using meshgrid
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    inputs = torch.stack([aa.flatten(), bb.flatten()], dim=1)  # (p^2, 2)
    labels = (inputs[:, 0] + inputs[:, 1]) % p  # (p^2,)

    return inputs, labels


def get_dataloaders(
    p: int,
    train_frac: float = 0.5,
    batch_size: int = 512,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Generate train and test dataloaders for modular addition.

    Args:
        p: Prime modulus
        train_frac: Fraction of data to use for training
        batch_size: Batch size for dataloaders
        seed: Random seed for reproducibility
        num_workers: Number of subprocesses to use for data loading
        pin_memory: If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them

    Returns:
        train_loader, test_loader
    """
    inputs, labels = generate_modular_addition_data(p)

    # Shuffle and split
    generator = torch.Generator().manual_seed(seed)
    n_samples = len(inputs)
    perm = torch.randperm(n_samples, generator=generator)
    n_train = int(train_frac * n_samples)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_dataset = TensorDataset(inputs[train_idx], labels[train_idx])
    test_dataset = TensorDataset(inputs[test_idx], labels[test_idx])

    persistent_workers = True if num_workers > 0 else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, test_loader
