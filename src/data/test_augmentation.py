from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from data.augment import get_train_augmentation


def main() -> None:
    image_path = next(
        Path("data/processed").glob(
            "person_*/image_*.jpg"
        )
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    augmentation = get_train_augmentation()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 7),
    )

    axes = axes.flatten()

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i in range(1, 6):
        augmented = augmentation(
            image=image
        )["image"]

        axes[i].imshow(augmented)
        axes[i].set_title(f"Augmented {i}")
        axes[i].axis("off")

    plt.tight_layout()

    output = Path(
        "outputs/plots/augmentation_examples.png"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()