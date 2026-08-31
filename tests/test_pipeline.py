from pathlib import Path

import cv2

from src.pipeline.recognize_image import process_image
from src.detection.detector import FaceDetector
from src.recognition.embedder import FaceEmbedder
from src.recognition.recognizer import FaceRecognizer


EMBEDDINGS_PATH = Path(
    "data/embeddings/train_embeddings.npz"
)

NORMAL_IMAGE = Path(
    "data/input/grppictestBM.jpg"
)

SMALL_IMAGE = Path(
    "data/input/test.jpg"
)


def create_models():
    """Create detector, embedder and recognizer."""

    detector = FaceDetector(
        det_size=(640, 640),
        det_thresh=0.5,
    )

    embedder = FaceEmbedder()

    recognizer = FaceRecognizer(
        EMBEDDINGS_PATH,
        threshold=0.45,
    )

    return detector, embedder, recognizer


def test_process_normal_image():
    """Normal multi-face image should use SCRFD detection."""

    image = cv2.imread(str(NORMAL_IMAGE))

    assert image is not None
    assert min(image.shape[:2]) >= 300

    detector, embedder, recognizer = create_models()

    output, results, used_crop_fallback = process_image(
        image,
        detector,
        embedder,
        recognizer,
    )

    assert output.shape == image.shape
    assert isinstance(results, list)
    assert used_crop_fallback is False

    # The test image is known to contain multiple faces.
    assert len(results) > 0


def test_process_small_face_crop():
    """Small cropped face should use ArcFace fallback."""

    image = cv2.imread(str(SMALL_IMAGE))

    assert image is not None
    assert min(image.shape[:2]) < 300

    detector, embedder, recognizer = create_models()

    output, results, used_crop_fallback = process_image(
        image,
        detector,
        embedder,
        recognizer,
    )

    assert output.shape == image.shape
    assert isinstance(results, list)
    assert used_crop_fallback is True

    assert len(results) == 1
    assert results[0]["identity"] != ""
    assert results[0]["similarity"] >= 0.0