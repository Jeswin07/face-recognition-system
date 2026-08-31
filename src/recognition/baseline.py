from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_EMBEDDINGS = PROJECT_ROOT / "data" / "embeddings" / "train_embeddings.npz"
TEST_EMBEDDINGS = PROJECT_ROOT / "data" / "embeddings" / "test_embeddings.npz"


def load_embeddings(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load embeddings, labels, and identity names."""

    data = np.load(path)

    return (
        data["embeddings"].astype(np.float32),
        data["labels"].astype(np.int64),
        data["identities"],
    )


def cosine_similarity_matrix(
    queries: np.ndarray,
    references: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between every query and reference.

    Because ArcFace embeddings are L2-normalized, this is
    equivalent to a matrix multiplication.
    """

    query_norms = np.linalg.norm(
        queries,
        axis=1,
        keepdims=True,
    )

    reference_norms = np.linalg.norm(
        references,
        axis=1,
        keepdims=True,
    )

    queries_normalized = queries / np.clip(
        query_norms,
        1e-12,
        None,
    )

    references_normalized = references / np.clip(
        reference_norms,
        1e-12,
        None,
    )

    return queries_normalized @ references_normalized.T


def evaluate_nearest_neighbor() -> None:
    """Evaluate cosine-similarity nearest-neighbor recognition."""

    train_embeddings, train_labels, train_identities = load_embeddings(
        TRAIN_EMBEDDINGS
    )

    test_embeddings, test_labels, test_identities = load_embeddings(
        TEST_EMBEDDINGS
    )

    print("Train embeddings:", train_embeddings.shape)
    print("Test embeddings:", test_embeddings.shape)

    similarity = cosine_similarity_matrix(
        test_embeddings,
        train_embeddings,
    )

    nearest_indices = np.argmax(
        similarity,
        axis=1,
    )

    predicted_labels = train_labels[nearest_indices]

    accuracy = accuracy_score(
        test_labels,
        predicted_labels,
    )

    print("\nRecognition Results")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Confidence-style similarity statistics.
    best_scores = similarity[
        np.arange(len(test_embeddings)),
        nearest_indices,
    ]

    correct = predicted_labels == test_labels

    print("\nSimilarity Statistics")
    print("-" * 40)
    print(f"Mean best similarity: {best_scores.mean():.4f}")
    print(f"Correct mean similarity: {best_scores[correct].mean():.4f}")

    if (~correct).any():
        print(
            f"Incorrect mean similarity: "
            f"{best_scores[~correct].mean():.4f}"
        )

    print("\nExample Predictions")
    print("-" * 40)

    for i in range(min(10, len(test_embeddings))):
        predicted_index = nearest_indices[i]

        print(
            f"Actual: {test_identities[i]:<30} | "
            f"Predicted: {train_identities[predicted_index]:<30} | "
            f"Similarity: {best_scores[i]:.4f}"
        )

    cm = confusion_matrix(
        test_labels,
        predicted_labels,
    )

    print("\nConfusion matrix shape:", cm.shape)


if __name__ == "__main__":
    evaluate_nearest_neighbor()