import albumentations as A


def get_train_augmentation() -> A.Compose:
    """Return augmentations used only for training images."""

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),

            A.Affine(
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),

            A.GaussNoise(
                std_range=(0.01, 0.03),
                p=0.2,
            ),
        ]
    )


def get_test_augmentation() -> A.Compose:
    """
    Return the test preprocessing pipeline.

    No random augmentation is applied to test images.
    """

    return A.Compose([])