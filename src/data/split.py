from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
TRAIN_RATIO = 0.8


def split_dataset(
    manifest_path: Path = Path(
        "data/processed/dataset_manifest.csv"
    ),
    output_dir: Path = Path("data/processed"),
) -> None:
    """Create a stratified 80/20 train/test split."""

    df = pd.read_csv(manifest_path)

    train_df, test_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        stratify=df["person_id"],
    )

    train_df = train_df.sort_values(
        ["person_id", "image_path"]
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        ["person_id", "image_path"]
    ).reset_index(drop=True)

    train_df.to_csv(
        output_dir / "train.csv",
        index=False,
    )

    test_df.to_csv(
        output_dir / "test.csv",
        index=False,
    )

    print("Dataset split complete.")
    print(f"Training images: {len(train_df)}")
    print(f"Testing images: {len(test_df)}")
    print(
        f"Train ratio: {len(train_df) / len(df):.2%}"
    )
    print(
        f"Test ratio: {len(test_df) / len(df):.2%}"
    )


if __name__ == "__main__":
    split_dataset()