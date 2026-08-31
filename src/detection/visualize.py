import cv2
import numpy as np

from pathlib import Path


def draw_detections(
    image: np.ndarray,
    faces: list[dict],
) -> np.ndarray:
    """Draw face bounding boxes and confidence scores."""

    output = image.copy()

    for face in faces:
        x1, y1, x2, y2 = face["bbox"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = f"Face {face['det_score']:.2f}"

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return output


def save_detection_result(
    image: np.ndarray,
    faces: list[dict],
    output_path: Path,
) -> None:
    """Draw and save detection results."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotated = draw_detections(
        image,
        faces,
    )

    if not cv2.imwrite(
        str(output_path),
        annotated,
    ):
        raise RuntimeError(
            f"Failed to save image: {output_path}"
        )