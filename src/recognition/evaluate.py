from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


ROOT = Path(__file__).resolve().parents[2]

TRAIN_EMBEDDINGS = ROOT / "data/embeddings/train_embeddings.npz"
TEST_EMBEDDINGS = ROOT / "data/embeddings/test_embeddings.npz"
LABEL_MAPPING = ROOT / "data/embeddings/label_mapping.csv"

OUTPUT_DIR = ROOT / "outputs/plots"


def cosine_similarity_matrix(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between every query and gallery embedding."""
    query_norm = query_embeddings / np.linalg.norm(
        query_embeddings, axis=1, keepdims=True
    )
    gallery_norm = gallery_embeddings / np.linalg.norm(
        gallery_embeddings, axis=1, keepdims=True
    )

    return query_norm @ gallery_norm.T


def load_data():
    """Load train/test embeddings and identity metadata."""
    train = np.load(TRAIN_EMBEDDINGS)
    test = np.load(TEST_EMBEDDINGS)
    mapping = pd.read_csv(LABEL_MAPPING)

    return train, test, mapping


def evaluate_thresholds(
    similarities: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
):
    """Evaluate recognition accuracy across similarity thresholds."""

    best_threshold = None
    best_accuracy = -1.0

    results = []

    for threshold in np.arange(0.20, 0.81, 0.01):
        threshold_predictions = predicted_labels.copy()

        best_scores = similarities.max(axis=1)
        unknown_mask = best_scores < threshold

        # -1 represents Unknown.
        threshold_predictions[unknown_mask] = -1

        # For the closed-set evaluation, Unknown is counted as incorrect.
        accuracy = np.mean(threshold_predictions == true_labels)

        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "unknown_count": int(unknown_mask.sum()),
            }
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return pd.DataFrame(results), best_threshold, best_accuracy


def plot_threshold_accuracy(results: pd.DataFrame):
    """Save accuracy versus similarity threshold plot."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(results["threshold"], results["accuracy"])
    plt.xlabel("Cosine Similarity Threshold")
    plt.ylabel("Recognition Accuracy")
    plt.title("Recognition Accuracy vs Similarity Threshold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "accuracy_vs_threshold.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def plot_similarity_distribution(
    best_scores: np.ndarray,
    correct_mask: np.ndarray,
):
    """Save similarity distributions for correct and incorrect predictions."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.hist(
        best_scores[correct_mask],
        bins=25,
        alpha=0.7,
        label="Correct",
    )

    plt.hist(
        best_scores[~correct_mask],
        bins=25,
        alpha=0.7,
        label="Incorrect",
    )

    plt.xlabel("Best Cosine Similarity")
    plt.ylabel("Number of Samples")
    plt.title("Recognition Similarity Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "similarity_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    train, test, mapping = load_data()

    train_embeddings = train["embeddings"]
    train_labels = train["labels"]

    test_embeddings = test["embeddings"]
    test_labels = test["labels"]

    print("=" * 60)
    print("Face Recognition Evaluation")
    print("=" * 60)

    print(f"Train embeddings : {train_embeddings.shape}")
    print(f"Test embeddings  : {test_embeddings.shape}")
    print(f"Identities       : {len(mapping)}")

    # Compare every test embedding against every training embedding.
    similarities = cosine_similarity_matrix(
        test_embeddings,
        train_embeddings,
    )

    best_indices = np.argmax(similarities, axis=1)
    best_scores = similarities[
        np.arange(len(test_embeddings)),
        best_indices,
    ]

    predicted_labels = train_labels[best_indices]

    correct_mask = predicted_labels == test_labels

    accuracy = accuracy_score(
        test_labels,
        predicted_labels,
    )

    print("\nClosed-set Recognition")
    print("-" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Correct: {correct_mask.sum()}/{len(test_labels)}")

    print("\nSimilarity Statistics")
    print("-" * 60)
    print(f"Mean: {best_scores.mean():.4f}")
    print(f"Min: {best_scores.min():.4f}")
    print(f"Max: {best_scores.max():.4f}")

    print(
        f"Correct mean:   "
        f"{best_scores[correct_mask].mean():.4f}"
    )

    if (~correct_mask).any():
        print(
            f"Incorrect mean: "
            f"{best_scores[~correct_mask].mean():.4f}"
        )

    # Per-identity accuracy.
    print("\nPer-Identity Accuracy")
    print("-" * 60)

    cm = confusion_matrix(
        test_labels,
        predicted_labels,
        labels=np.arange(len(mapping)),
    )

    rows = []

    for label in range(len(mapping)):
        total = cm[label].sum()

        if total == 0:
            continue

        class_accuracy = cm[label, label] / total

        rows.append(
            {
                "label": label,
                "identity": mapping.loc[
                    mapping["label"] == label,
                    "identity",
                ].iloc[0],
                "correct": cm[label, label],
                "total": total,
                "accuracy": class_accuracy,
            }
        )

    per_identity = pd.DataFrame(rows)

    print(
        per_identity.sort_values("accuracy").head(10).to_string(
            index=False
        )
    )

    # Threshold analysis.
    results, best_threshold, best_threshold_accuracy = (
        evaluate_thresholds(
            similarities,
            test_labels,
            predicted_labels,
        )
    )

    print("\nThreshold Analysis")
    print("-" * 60)
    print(f"Best threshold: {best_threshold:.2f}")
    print(
        f"Best accuracy:  "
        f"{best_threshold_accuracy:.4f} "
        f"({best_threshold_accuracy * 100:.2f}%)"
    )

    print("\nThreshold Results")
    print("-" * 60)
    print(
        results[
            results["threshold"].isin(
                [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
            )
        ].to_string(index=False)
    )

    # Save detailed results.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_identity.to_csv(
        OUTPUT_DIR / "per_identity_accuracy.csv",
        index=False,
    )

    results.to_csv(
        OUTPUT_DIR / "threshold_results.csv",
        index=False,
    )

    plot_threshold_accuracy(results)

    plot_similarity_distribution(
        best_scores,
        correct_mask,
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()