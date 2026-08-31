from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_people


NUM_IDENTITIES = 50
IMAGES_PER_IDENTITY = 12
RANDOM_SEED = 42


def build_dataset(
    output_dir: Path = Path("data/processed"),
) -> None:
    """Create a balanced subset of LFW for our project."""

    rng = np.random.default_rng(RANDOM_SEED)

    dataset = fetch_lfw_people(
        data_home="data/raw/lfw",
        min_faces_per_person=IMAGES_PER_IDENTITY,
        color=True,
        resize=1.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group image indices by identity.
    identity_indices: dict[int, list[int]] = {}

    for index, target in enumerate(dataset.target):
        identity_indices.setdefault(int(target), []).append(index)

    # Select identities reproducibly.
    available_identities = np.array(
        sorted(identity_indices.keys())
    )

    selected_identities = rng.choice(
        available_identities,
        size=NUM_IDENTITIES,
        replace=False,
    )

    selected_identities = sorted(
        selected_identities.tolist()
    )

    records = []

    for new_identity_id, original_identity_id in enumerate(
        selected_identities
    ):
        identity_name = str(
            dataset.target_names[original_identity_id]
        )

        indices = np.array(
            identity_indices[original_identity_id]
        )

        selected_images = rng.choice(
            indices,
            size=IMAGES_PER_IDENTITY,
            replace=False,
        )

        person_dir = output_dir / f"person_{new_identity_id:03d}"
        person_dir.mkdir(parents=True, exist_ok=True)

        for image_number, image_index in enumerate(
            sorted(selected_images)
        ):
            image = dataset.images[image_index]

            # LFW images are RGB when color=True.
            image_uint8 = np.clip(
                image * 255.0,
                0,
                255,
            ).astype(np.uint8)

            image_path = (
                person_dir
                / f"image_{image_number:03d}.jpg"
            )

            from PIL import Image

            Image.fromarray(image_uint8).save(
                image_path,
                quality=95,
            )

            records.append(
                {
                    "person_id": f"person_{new_identity_id:03d}",
                    "identity": identity_name,
                    "image_path": str(image_path),
                    "source_index": int(image_index),
                }
            )

    manifest = pd.DataFrame(records)

    manifest.to_csv(
        output_dir / "dataset_manifest.csv",
        index=False,
    )

    print("Dataset created successfully.")
    print(f"Identities: {NUM_IDENTITIES}")
    print(f"Images: {len(manifest)}")
    print(f"Manifest: {output_dir / 'dataset_manifest.csv'}")


if __name__ == "__main__":
    build_dataset()