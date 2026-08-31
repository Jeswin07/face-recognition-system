from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    train_dir: Path = Path("data/train")
    test_dir: Path = Path("data/test")

    num_identities: int = 50
    images_per_identity: int = 12

    train_ratio: float = 0.8
    random_seed: int = 42