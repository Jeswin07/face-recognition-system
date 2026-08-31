from pathlib import Path

import numpy as np

from src.recognition.recognizer import FaceRecognizer


def main() -> None:
    print("=" * 60)
    print("Face Recognizer Test")
    print("=" * 60)

    train_path = Path("data/embeddings/train_embeddings.npz")
    test_path = Path("data/embeddings/test_embeddings.npz")

    # ---------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------
    test_data = np.load(test_path)

    test_embeddings = test_data["embeddings"]
    test_labels = test_data["labels"]
    test_identities = test_data["identities"]

    print(f"Train embeddings : {np.load(train_path)['embeddings'].shape[0]}")
    print(f"Test embeddings  : {len(test_embeddings)}")
    print(f"Identities       : {len(np.unique(test_labels))}")
    print("Threshold        : 0.45")

    # ---------------------------------------------------------
    # Initialize recognizer with training gallery
    # ---------------------------------------------------------
    recognizer = FaceRecognizer(
        embeddings_path=train_path,
        threshold=0.45,
    )

    # ---------------------------------------------------------
    # Recognition
    # ---------------------------------------------------------
    predictions = []

    for embedding in test_embeddings:
        result = recognizer.recognize_embedding(embedding)
        predictions.append(result)

    # ---------------------------------------------------------
    # Closed-set accuracy
    #
    # For accuracy, compare the predicted label against
    # the ground-truth label directly.
    # ---------------------------------------------------------
    predicted_labels = np.array(
        [result["label"] for result in predictions]
    )

    correct = predicted_labels == test_labels
    accuracy = float(correct.mean())

    print()
    print("Accuracy         :", f"{accuracy:.4f}")
    print("Accuracy         :", f"{accuracy * 100:.2f}%")

    # ---------------------------------------------------------
    # Example predictions
    # ---------------------------------------------------------
    print()
    print("Example predictions")
    print("-" * 60)

    for i in range(min(10, len(predictions))):
        result = predictions[i]

        actual = str(test_identities[i])
        predicted = result["identity"]
        similarity = result["similarity"]

        print(
            f"Actual: {actual:<30} | "
            f"Predicted: {predicted:<30} | "
            f"Similarity: {similarity:.4f}"
        )


if __name__ == "__main__":
    main()