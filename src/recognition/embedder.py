from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from insightface.model_zoo import get_model


class FaceEmbedder:
    """
    Extract 512-dimensional face embeddings using pretrained ArcFace.

    The input is expected to be a single face crop. Detection is intentionally
    handled by a separate component in the production pipeline.
    """

    MODEL_NAME = "w600k_r50.onnx"
    MODEL_DIR = Path.home() / ".insightface" / "models" / "buffalo_l"

    def __init__(self, device: str = "cpu") -> None:
        model_path = self.MODEL_DIR / self.MODEL_NAME

        if not model_path.exists():
            raise FileNotFoundError(
                f"ArcFace model not found: {model_path}. "
                "Initialize InsightFace/buffalo_l first."
            )

        # InsightFace's ONNX models currently run through ONNX Runtime.
        # Our environment has CPUExecutionProvider available.
        providers = ["CPUExecutionProvider"]

        self.model = get_model(
            str(model_path),
            providers=providers,
        )

        # ctx_id=-1 explicitly selects CPU.
        self.model.prepare(ctx_id=-1)

    def embed(self, image: np.ndarray) -> np.ndarray:
        """
        Extract a normalized ArcFace embedding.

        Parameters
        ----------
        image:
            BGR OpenCV image containing one face.

        Returns
        -------
        np.ndarray
            L2-normalized embedding with shape (512,).
        """
        if image is None:
            raise ValueError("Input image is None.")

        if image.ndim != 3:
            raise ValueError(
                f"Expected HxWxC image, got shape {image.shape}."
            )

        if image.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel BGR image, got shape {image.shape}."
            )

        embedding = self.model.get_feat(image)[0].astype(np.float32)

        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError("ArcFace returned a zero-norm embedding.")

        return embedding / norm

    def embed_path(self, image_path: str | Path) -> np.ndarray:
        """Load an image from disk and extract its embedding."""
        image_path = Path(image_path)

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        return self.embed(image)