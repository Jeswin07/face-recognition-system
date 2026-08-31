from pathlib import Path

from src.detection.detector import FaceDetector


def main():
    image_path = next(
        Path("data/processed").glob("person_*/image_*.jpg")
    )

    detector = FaceDetector()

    faces = detector.detect_path(image_path)

    print("=" * 60)
    print("Face Detection Test")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Faces detected: {len(faces)}")

    for i, face in enumerate(faces, start=1):
        print(f"\nFace {i}")
        print(f"  Bounding box : {face['bbox']}")
        print(f"  Confidence   : {face['det_score']:.4f}")

        if face["landmarks"] is not None:
            print(
                f"  Landmarks    : "
                f"{len(face['landmarks'])} points"
            )


if __name__ == "__main__":
    main()