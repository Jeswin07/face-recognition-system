from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceDetector:
    """Face detector based on InsightFace SCRFD."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
    ):
        self.det_size = det_size
        self.det_thresh = det_thresh

        self.app = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"],
        )

        self.app.prepare(
            ctx_id=0,
            det_thresh=det_thresh,
            det_size=det_size,
        )

    @staticmethod
    def _resize_for_detection(
        image: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Upscale very small images before face detection.

        SCRFD works much better when the input image provides
        enough pixels for the face detector.
        """

        h, w = image.shape[:2]
        min_dimension = min(h, w)

        # Normal-sized images need no modification.
        if min_dimension >= 640:
            return image, 1.0

        # Scale the smallest dimension to approximately 640 px.
        scale = 640.0 / min_dimension

        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        return resized, scale

    def detect(self, image: np.ndarray) -> list[dict]:
        """Detect faces in a BGR OpenCV image."""

        if image is None:
            raise ValueError("Input image is None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Input image must be a NumPy array."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected BGR image with shape (H, W, 3), "
                f"got {image.shape}"
            )

        detection_image, scale = self._resize_for_detection(
            image
        )

        faces = self.app.get(detection_image)

        results = []

        for face in faces:
            bbox = face.bbox.astype(float) / scale

            # Keep coordinates inside original image.
            h, w = image.shape[:2]

            bbox[0] = np.clip(bbox[0], 0, w - 1)
            bbox[1] = np.clip(bbox[1], 0, h - 1)
            bbox[2] = np.clip(bbox[2], 0, w - 1)
            bbox[3] = np.clip(bbox[3], 0, h - 1)

            landmarks = None

            if face.kps is not None:
                landmarks = (
                    face.kps.astype(float) / scale
                ).tolist()

            results.append(
                {
                    "bbox": bbox.astype(int).tolist(),
                    "det_score": float(face.det_score),
                    "landmarks": landmarks,
                    "embedding": (
                        face.embedding.astype(np.float32)
                        if hasattr(face, "embedding")
                        and face.embedding is not None
                        else None
                    ),
                }
            )

        return results

    def detect_path(
        self,
        image_path: str | Path,
    ) -> list[dict]:
        """Detect faces in an image file."""

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"OpenCV could not read image: {image_path}"
            )

        return self.detect(image)