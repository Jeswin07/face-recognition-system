from pathlib import Path

from sklearn.datasets import fetch_lfw_people


def download_lfw(
    data_home: Path = Path("data/raw/lfw"),
    min_faces_per_person: int = 12,
):
    """
    Download and load the Labeled Faces in the Wild dataset.

    Only identities with at least `min_faces_per_person`
    images are retained.
    """

    data_home.mkdir(parents=True, exist_ok=True)

    dataset = fetch_lfw_people(
        data_home=str(data_home),
        min_faces_per_person=min_faces_per_person,
        color=True,
        resize=1.0,
    )

    return dataset


if __name__ == "__main__":
    dataset = download_lfw()

    print(f"Images: {len(dataset.images)}")
    print(f"Identities: {len(dataset.target_names)}")
    print(f"Image shape: {dataset.images.shape[1:]}")