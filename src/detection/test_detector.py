from pathlib import Path

import cv2

from src.detection.detector import FaceDetector
from src.detection.visualize import save_detection_result

def main() -> None:
    detector = FaceDetector()

    image_path = next(
        Path("data/processed").glob(
            "person_*/image_*.jpg"
        )
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Could not read {image_path}"
        )

    faces = detector.detect(image)

    print(f"Image: {image_path}")
    print(f"Faces detected: {len(faces)}")

    save_detection_result(
        image,
        faces,
        Path("outputs/detection/single_face.jpg"),
    )

    for index, face in enumerate(faces):
        print(
            f"Face {index}: "
            f"bbox={face.bbox.tolist()}, "
            f"confidence={face.confidence:.4f}"
        )


if __name__ == "__main__":
    main()