from pathlib import Path

import cv2
import pytest

from src.detection.detector import FaceDetector


TEST_IMAGE = Path("data/input/test.jpg")


@pytest.fixture(scope="module")
def detector():
    return FaceDetector(
        det_size=(640, 640),
        det_thresh=0.5,
    )


def test_detector_initializes(detector):
    assert detector is not None
    assert detector.app is not None


def test_detect_rejects_none(detector):
    with pytest.raises(ValueError):
        detector.detect(None)


def test_detect_rejects_invalid_type(detector):
    with pytest.raises(TypeError):
        detector.detect("not an image")


def test_detect_rejects_grayscale_image(detector):
    image = cv2.imread(str(TEST_IMAGE))

    if image is None:
        pytest.skip("Test image not available.")

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    with pytest.raises(ValueError):
        detector.detect(grayscale)


def test_detect_real_image(detector):
    if not TEST_IMAGE.exists():
        pytest.skip("Test image not available.")

    image = cv2.imread(str(TEST_IMAGE))

    if image is None:
        pytest.skip("Could not read test image.")

    faces = detector.detect(image)

    assert isinstance(faces, list)

    for face in faces:
        assert "bbox" in face
        assert "det_score" in face
        assert "landmarks" in face
        assert "embedding" in face

        assert len(face["bbox"]) == 4
        assert 0.0 <= face["det_score"] <= 1.0

        if face["embedding"] is not None:
            assert face["embedding"].shape == (512,)


def test_detect_path_missing_file(detector):
    with pytest.raises(FileNotFoundError):
        detector.detect_path(
            "data/does_not_exist.jpg"
        )