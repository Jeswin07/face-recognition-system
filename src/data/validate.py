from pathlib import Path

from PIL import Image


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def is_valid_image(path: Path) -> bool:
    """Return True when the file can be opened as a valid image."""
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return False

    try:
        with Image.open(path) as image:
            image.verify()

        return True

    except (OSError, ValueError):
        return False