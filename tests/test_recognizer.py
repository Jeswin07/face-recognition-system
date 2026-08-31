import numpy as np

from src.recognition.recognizer import FaceRecognizer


EMBEDDINGS_PATH = "data/embeddings/train_embeddings.npz"


def test_recognizer_loads_embeddings():
    recognizer = FaceRecognizer(
        EMBEDDINGS_PATH,
        threshold=0.45,
    )

    assert recognizer.embeddings is not None
    assert len(recognizer.embeddings) == 480


def test_recognizer_returns_expected_keys():
    recognizer = FaceRecognizer(
        EMBEDDINGS_PATH,
        threshold=0.45,
    )

    embedding = recognizer.embeddings[0]

    result = recognizer.recognize_embedding(embedding)

    assert "identity" in result
    assert "person_id" in result
    assert "label" in result
    assert "similarity" in result
    assert "recognized" in result


def test_same_training_embedding_is_recognized():
    recognizer = FaceRecognizer(
        EMBEDDINGS_PATH,
        threshold=0.45,
    )

    embedding = recognizer.embeddings[0]

    result = recognizer.recognize_embedding(embedding)

    assert result["recognized"] is True
    assert result["identity"] != "Unknown"
    assert result["similarity"] >= 0.45


def test_random_embedding_is_unknown():
    recognizer = FaceRecognizer(
        EMBEDDINGS_PATH,
        threshold=0.45,
    )

    rng = np.random.default_rng(42)
    embedding = rng.normal(size=512).astype(np.float32)
    embedding /= np.linalg.norm(embedding)

    result = recognizer.recognize_embedding(embedding)

    assert result["identity"] == "Unknown"
    assert result["recognized"] is False


def test_threshold_changes_recognition():
    recognizer = FaceRecognizer(
        EMBEDDINGS_PATH,
        threshold=0.45,
    )

    embedding = recognizer.embeddings[0]

    result = recognizer.recognize_embedding(embedding)

    similarity = result["similarity"]

    strict_recognizer = FaceRecognizer(
        EMBEDDINGS_PATH,
        threshold=similarity + 0.01,
    )

    strict_result = strict_recognizer.recognize_embedding(
        embedding
    )

    assert strict_result["recognized"] is False
    assert strict_result["identity"] == "Unknown"