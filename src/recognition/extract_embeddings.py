from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.recognition.embedder import FaceEmbedder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"


def build_label_mapping(df: pd.DataFrame) -> dict[str, int]:
    """Create a stable integer label for every identity."""

    person_ids = sorted(df["person_id"].unique())

    return {
        person_id: label
        for label, person_id in enumerate(person_ids)
    }


def extract_split(
    embedder: FaceEmbedder,
    csv_path: Path,
    split_name: str,
    label_mapping: dict[str, int],
) -> None:
    """Extract and save ArcFace embeddings for one dataset split."""

    df = pd.read_csv(csv_path)

    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    person_ids: list[str] = []
    identities: list[str] = []
    image_paths: list[str] = []

    failures: list[tuple[str, str]] = []

    total = len(df)
    start_time = time.perf_counter()

    print(f"\n{'=' * 60}")
    print(f"Extracting {split_name} embeddings")
    print(f"Images: {total}")
    print(f"{'=' * 60}")

    for index, row in df.iterrows():
        image_path = PROJECT_ROOT / row["image_path"]
        person_id = str(row["person_id"])

        try:
            embedding = embedder.embed_path(image_path)

            embeddings.append(embedding)
            labels.append(label_mapping[person_id])
            person_ids.append(person_id)
            identities.append(str(row["identity"]))
            image_paths.append(str(row["image_path"]))

        except Exception as exc:
            failures.append(
                (str(row["image_path"]), str(exc))
            )

        if (index + 1) % 50 == 0 or index + 1 == total:
            print(f"Progress: {index + 1}/{total}")

    elapsed = time.perf_counter() - start_time

    if not embeddings:
        raise RuntimeError(
            f"No embeddings were extracted for {split_name}."
        )

    embedding_array = np.stack(embeddings).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)

    output_path = OUTPUT_DIR / f"{split_name}_embeddings.npz"

    np.savez_compressed(
        output_path,
        embeddings=embedding_array,
        labels=label_array,
        person_ids=np.asarray(person_ids),
        identities=np.asarray(identities),
        image_paths=np.asarray(image_paths),
    )

    print(f"\nSaved: {output_path}")
    print(f"Embedding shape: {embedding_array.shape}")
    print(f"Labels shape: {label_array.shape}")
    print(f"Identities: {len(np.unique(person_ids))}")
    print(f"Failures: {len(failures)}")
    print(f"Time: {elapsed:.2f}s")

    if failures:
        print("\nFailed images:")
        for path, error in failures:
            print(f"  {path}: {error}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Build mapping from the training set only.
    # The test set must use exactly the same label space.
    label_mapping = build_label_mapping(train_df)

    # Sanity check: train and test must contain the same identities.
    train_ids = set(train_df["person_id"])
    test_ids = set(test_df["person_id"])

    if train_ids != test_ids:
        missing_from_test = train_ids - test_ids
        missing_from_train = test_ids - train_ids

        raise ValueError(
            "Train/test identity mismatch.\n"
            f"Missing from test: {missing_from_test}\n"
            f"Missing from train: {missing_from_train}"
        )

    print(f"Number of identities: {len(label_mapping)}")

    print("Initializing ArcFace embedder...")
    embedder = FaceEmbedder()

    extract_split(
        embedder=embedder,
        csv_path=TRAIN_CSV,
        split_name="train",
        label_mapping=label_mapping,
    )

    extract_split(
        embedder=embedder,
        csv_path=TEST_CSV,
        split_name="test",
        label_mapping=label_mapping,
    )

    # Save the mapping so inference can translate
    # classifier labels back into identity names.
    mapping_path = OUTPUT_DIR / "label_mapping.csv"

    mapping_df = pd.DataFrame(
        [
            {
                "label": label,
                "person_id": person_id,
                "identity": train_df.loc[
                    train_df["person_id"] == person_id,
                    "identity",
                ].iloc[0],
            }
            for person_id, label in label_mapping.items()
        ]
    ).sort_values("label")

    mapping_df.to_csv(mapping_path, index=False)

    print(f"\nLabel mapping saved: {mapping_path}")
    print("\nEmbedding extraction complete.")


if __name__ == "__main__":
    main()