import albumentations as A


class Augmentations:
    """Class for image augmentations."""

    @staticmethod
    def get_train_augs():
        return A.Compose([
            A.Resize(320, 320),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ])

    @staticmethod
    def get_valid_augs():
        return A.Compose([
            A.Resize(320, 320)
        ])
