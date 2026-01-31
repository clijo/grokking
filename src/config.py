from dataclasses import dataclass, field


@dataclass
class GrokkingConfig:
    # Task
    p: int = field(default=113, metadata={"help": "Prime modulus"})

    # Model
    d_model: int = field(default=64, metadata={"help": "Embedding dimension"})
    nhead: int = field(default=4, metadata={"help": "Number of attention heads"})
    dim_feedforward: int = field(
        default=256, metadata={"help": "Feedforward dimension"}
    )
    dropout: float = field(default=0.0, metadata={"help": "Dropout rate"})

    # Training
    lr: float = field(default=1e-3, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=1.0, metadata={"help": "Weight decay"})
    epochs: int = field(default=2000, metadata={"help": "Number of epochs"})
    batch_size: int = field(default=1024, metadata={"help": "Batch size"})

    # Data
    train_frac: float = field(
        default=0.25, metadata={"help": "Fraction of data for training"}
    )
    seed: int = field(default=42, metadata={"help": "Random seed"})
    num_workers: int = field(default=4, metadata={"help": "DataLoader workers"})
    pin_memory: bool = field(
        default=True, metadata={"help": "Pin memory for DataLoader"}
    )

    # Logging
    log_every: int = field(default=100, metadata={"help": "Log every N epochs"})
    eval_every: int = field(default=100, metadata={"help": "Evaluate every N epochs"})
    wandb_project: str = field(
        default="grokking-modular-addition", metadata={"help": "WandB project name"}
    )
