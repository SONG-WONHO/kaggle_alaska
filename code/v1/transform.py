from albumentations.pytorch import ToTensor
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Normalize, Cutout, PadIfNeeded, RandomCrop, ToFloat
)
import cv2


def transform_v0(config):
    train_transform = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        Cutout(num_holes=4, max_h_size=4, max_w_size=4, p=0.5),
        Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=0.5),
        Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.5),
        Cutout(num_holes=16, max_h_size=8, max_w_size=8, p=0.5),
        Cutout(num_holes=32, max_h_size=8, max_w_size=8, p=0.5),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    test_transform = Compose([
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    return train_transform, test_transform


def transform_v1(config):
    train_transform = Compose([
        Compose([
            PadIfNeeded(min_height=532,
                        min_width=532,
                        p=1.0),
            RandomCrop(512, 512, p=1.0)
        ], p=0.5),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        Cutout(num_holes=4, max_h_size=4, max_w_size=4, p=0.5),
        Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=0.5),
        Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.5),
        Cutout(num_holes=16, max_h_size=8, max_w_size=8, p=0.5),
        Cutout(num_holes=32, max_h_size=8, max_w_size=8, p=0.5),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor(),
    ], p=1)

    test_transform = Compose([
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor(),
    ], p=1)

    return train_transform, test_transform


def get_transform(config):
    try:
        name = f"transform_v{config.transform_version}"
        f = globals().get(name)
        print(f"... Transform Info - {name}")
        return f(config)

    except TypeError:
        raise NotImplementedError("try another transform version ...")