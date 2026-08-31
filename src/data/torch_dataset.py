from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.augment import (
    get_test_augmentation,
    get_train_augmentation,
)


class FaceDataset(Dataset):
    """PyTorch dataset for face recognition images."""

    def __init__(
        self,
        csv_path: str | Path,
        training: bool = False,
    ) -> None:
        self.data = pd.read_csv(csv_path)
        self.training = training

        self.transform = (
            get_train_augmentation()
            if training
            else get_test_augmentation()
        )

        identities = sorted(
            self.data["person_id"].unique()
        )

        self.label_map = {
            identity: index
            for index, identity in enumerate(identities)
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:

        row = self.data.iloc[index]

        image_path = Path(row["image_path"])

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        transformed = self.transform(
            image=image
        )

        image = transformed["image"]

        image = torch.from_numpy(
            image.copy()
        ).permute(2, 0, 1)

        image = image.float() / 255.0

        label = self.label_map[
            row["person_id"]
        ]

        return image, label