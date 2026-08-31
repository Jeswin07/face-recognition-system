from pathlib import Path

import cv2
import numpy as np

from src.detection.detector import FaceDetector
from src.recognition.embedder import FaceEmbedder
from src.recognition.recognizer import FaceRecognizer


INPUT_PATH = Path("data/input/test.jpg")
OUTPUT_PATH = Path("outputs/results/recognized_test.jpg")
EMBEDDINGS_PATH = Path("data/embeddings/train_embeddings.npz")

THRESHOLD = 0.45
SMALL_IMAGE_THRESHOLD = 300


def draw_result(
    image: np.ndarray,
    bbox: list[int],
    identity: str,
    similarity: float,
    det_score: float | None,
) -> None:
    """Draw one face recognition result."""

    x1, y1, x2, y2 = bbox

    label = (
        f"Unknown ({similarity:.2f})"
        if identity == "Unknown"
        else f"{identity} ({similarity:.2f})"
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if det_score is not None:
        cv2.putText(
            image,
            f"det: {det_score:.2f}",
            (
                x1,
                min(y2 + 20, image.shape[0] - 5),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )


def process_image(
    image: np.ndarray,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    recognizer: FaceRecognizer,
) -> tuple[np.ndarray, list[dict], bool]:
    """
    Detect and recognize faces in an image.

    Normal image:
        SCRFD -> ArcFace embedding -> recognition

    Small image:
        Treat the entire image as an already-cropped face
        and generate its ArcFace embedding directly.

    Returns:
        output image,
        recognition results,
        whether small-image fallback was used.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected BGR image with shape (H, W, 3), "
            f"got {image.shape}"
        )

    detections = detector.detect(image)

    output = image.copy()
    results = []

    # ---------------------------------------------------------
    # CASE 1: SCRFD detected one or more faces.
    # ---------------------------------------------------------
    if detections:

        for index, detection in enumerate(
            detections,
            start=1,
        ):
            embedding = detection["embedding"]

            if embedding is None:
                continue

            recognition = recognizer.recognize_embedding(
                embedding
            )

            bbox = detection["bbox"]
            det_score = detection["det_score"]

            draw_result(
                image=output,
                bbox=bbox,
                identity=recognition["identity"],
                similarity=recognition["similarity"],
                det_score=det_score,
            )

            results.append(
                {
                    "face": index,
                    "identity": recognition["identity"],
                    "similarity": recognition["similarity"],
                    "detection": det_score,
                }
            )

        return output, results, False

    # ---------------------------------------------------------
    # CASE 2: Small image is an already-cropped face.
    # ---------------------------------------------------------
    height, width = image.shape[:2]

    if min(height, width) < SMALL_IMAGE_THRESHOLD:

        embedding = embedder.embed(image)

        recognition = recognizer.recognize_embedding(
            embedding
        )

        bbox = [
            0,
            0,
            width - 1,
            height - 1,
        ]

        draw_result(
            image=output,
            bbox=bbox,
            identity=recognition["identity"],
            similarity=recognition["similarity"],
            det_score=None,
        )

        results.append(
            {
                "face": 1,
                "identity": recognition["identity"],
                "similarity": recognition["similarity"],
                "detection": None,
            }
        )

        return output, results, True

    # ---------------------------------------------------------
    # CASE 3: Normal image with no detected faces.
    # ---------------------------------------------------------
    return output, results, False


def recognize_image(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
):
    """Run the complete face detection and recognition pipeline."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {input_path}"
        )

    image = cv2.imread(str(input_path))

    if image is None:
        raise ValueError(
            f"Could not read input image: {input_path}"
        )

    print("=" * 60)
    print("End-to-End Face Detection + Recognition")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Image shape: {image.shape}")
    print()

    print("Initializing face detector...")

    detector = FaceDetector(
        det_size=(640, 640),
        det_thresh=0.5,
    )

    print("Initializing face embedder...")

    embedder = FaceEmbedder()

    print("Initializing face recognizer...")

    recognizer = FaceRecognizer(
        EMBEDDINGS_PATH,
        threshold=THRESHOLD,
    )

    print()
    print("Detecting and recognizing faces...")

    output, results, used_crop_fallback = process_image(
        image,
        detector,
        embedder,
        recognizer,
    )

    print(f"Faces recognized/detected: {len(results)}")

    if used_crop_fallback:
        print(
            "Small image fallback was used."
        )

    print()

    for result in results:
        print(
            f"Face {result['face']:2d} | "
            f"{result['identity']:<30} | "
            f"similarity: {result['similarity']:.4f} | "
            f"det: {result['detection']}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(output_path),
        output,
    ):
        raise IOError(
            f"Failed to save output image: {output_path}"
        )

    print()
    print("=" * 60)
    print(f"Saved annotated image: {output_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    recognize_image()