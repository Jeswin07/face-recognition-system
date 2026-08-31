from collections import Counter
from pathlib import Path

from .validate import is_valid_image


def collect_identity_images(
    root: Path,
) -> dict[str, list[Path]]:
    """Collect valid images grouped by identity."""

    identities: dict[str, list[Path]] = {}

    for identity_dir in sorted(root.iterdir()):
        if not identity_dir.is_dir():
            continue

        images = [
            path
            for path in sorted(identity_dir.iterdir())
            if is_valid_image(path)
        ]

        if images:
            identities[identity_dir.name] = images

    return identities


def dataset_summary(
    identities: dict[str, list[Path]],
) -> dict:
    """Return basic dataset statistics."""

    counts = Counter(
        len(images)
        for images in identities.values()
    )

    total_images = sum(counts.elements())

    return {
        "identities": len(identities),
        "total_images": total_images,
        "images_per_identity": dict(sorted(counts.items())),
        "min_images_per_identity": (
            min(counts) if counts else 0
        ),
        "max_images_per_identity": (
            max(counts) if counts else 0
        ),
    }