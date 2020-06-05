from albumentations.pytorch import ToTensor
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Normalize, Cutout, PadIfNeeded, RandomCrop, ToFloat,
    RandomGridShuffle, ChannelShuffle, GridDropout, OneOf
)
import cv2


def transform_v0(config):
    """ YH`s Basic Transform - VFlip, Hflip, Cutout, Normalize

    :param config: CFG
    :return: (train transform, test transform)
    """
    train_transform = Compose([
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    return train_transform


def get_transform(config):
    try:
        name = f"transform_v{config.transform_version}"
        f = globals().get(name)
        print(f"... Transform Info - {name}")
        return f(config)

    except TypeError:
        raise NotImplementedError("try another transform version ...")