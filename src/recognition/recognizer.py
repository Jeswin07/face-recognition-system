from pathlib import Path

import numpy as np
import pandas as pd


class FaceRecognizer:
    """Match face embeddings against a known identity gallery."""

    def __init__(
        self,
        embeddings_path: str | Path,
        threshold: float = 0.45,
    ):
        self.embeddings_path = Path(embeddings_path)
        self.threshold = threshold

        self._load_gallery()

    def _load_gallery(self) -> None:
        data = np.load(self.embeddings_path)

        self.embeddings = data["embeddings"].astype(np.float32)
        self.labels = data["labels"].astype(np.int64)
        self.person_ids = data["person_ids"]
        self.identities = data["identities"]

        # Safety: embeddings should already be normalized,
        # but normalize again so matching is deterministic.
        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True,
        )

        self.embeddings = self.embeddings / np.maximum(
            norms,
            1e-12,
        )

        self._label_to_identity = {}

        for label, identity in zip(
            self.labels,
            self.identities,
        ):
            self._label_to_identity[int(label)] = str(identity)

    def recognize_embedding(
        self,
        embedding: np.ndarray,
    ) -> dict:
        """Recognize a single normalized or unnormalized embedding."""

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if embedding.ndim != 1:
            raise ValueError(
                f"Expected 1D embedding, got shape {embedding.shape}"
            )

        if embedding.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embeddings.shape[1]}, "
                f"got {embedding.shape[0]}"
            )

        norm = np.linalg.norm(embedding)

        if norm < 1e-12:
            raise ValueError("Cannot recognize a zero embedding.")

        embedding = embedding / norm

        similarities = self.embeddings @ embedding

        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])

        label = int(self.labels[best_index])
        person_id = str(self.person_ids[best_index])

        if best_similarity >= self.threshold:
            identity = self._label_to_identity[label]
            recognized = True
        else:
            identity = "Unknown"
            recognized = False

        return {
            "identity": identity,
            "person_id": person_id,
            "label": label,
            "similarity": best_similarity,
            "recognized": recognized,
        }

    def recognize_embeddings(
        self,
        embeddings: np.ndarray,
    ) -> list[dict]:
        """Recognize multiple embeddings."""

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings, got shape {embeddings.shape}"
            )

        return [
            self.recognize_embedding(embedding)
            for embedding in embeddings
        ]

    def set_threshold(self, threshold: float) -> None:
        """Update the Unknown rejection threshold."""

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        self.threshold = threshold

    def __len__(self) -> int:
        """Return number of stored gallery embeddings."""

        return len(self.embeddings)