import numpy as np

from src.recognition.recognizer import FaceRecognizer


EMBEDDINGS_PATH = "data/embeddings/train_embeddings.npz"


def test_recognizer_loads_embeddings():
    recognizer = FaceRecognizer(
        EMBEDDINGS_PATH,
        threshold=0.45,
    )

    assert len(recognizer.embeddings) == 480
    assert len(recognizer.labels) == 480
    assert len(set(recognizer.identities)) == 50


def test_known_embedding_is_recognized():
    recognizer = FaceRecognizer(
        EMBEDDINGS_PATH,
        threshold=0.45,
    )

    embedding = recognizer.embeddings[0]

    result = recognizer.recognize_embedding(embedding)

    assert result["recognized"] is True
    assert result["identity"] != "Unknown"
    assert result["similarity"] >= 0.45


def test_unknown_embedding_is_rejected():
    recognizer = FaceRecognizer(
        EMBEDDINGS_PATH,
        threshold=0.45,
    )

    rng = np.random.default_rng(42)
    embedding = rng.normal(
        size=recognizer.embeddings[0].shape
    ).astype(np.float32)

    result = recognizer.recognize_embedding(embedding)

    assert result["identity"] == "Unknown"
    assert result["recognized"] is False


def test_embedding_dimension():
    recognizer = FaceRecognizer(
        EMBEDDINGS_PATH,
        threshold=0.45,
    )

    embedding = recognizer.embeddings[0]

    assert embedding.shape == (512,)


def test_recognition_result_structure():
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

    assert isinstance(result["similarity"], float)
    assert isinstance(result["recognized"], bool)