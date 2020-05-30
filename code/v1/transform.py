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
    """ v0 + Random 512 Crop after Padding 10px each border

    :param config: CFG
    :return: (train transform, test transform)
    """

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


def transform_v2(config):
    """ v0 + Random 256 Crop

    :param config: CFG
    :return: (train transform, test transform)
    """
    train_transform = Compose([
        RandomCrop(256, 256, p=1.0),
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
        RandomCrop(256, 256, p=1.0),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor(),
    ], p=1)

    return train_transform, test_transform


def transform_v3(config):
    """ Grid Shuffle, Vflip, Hflip, Cutout, Normalize

    :param config: CFG
    :return: (train transform, test transform)
    """
    train_transform = Compose([
        RandomGridShuffle((4, 4), p=0.5),
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


def transform_v4(config):
    """ ChannelShuffle, VFlip, Hflip, Cutout, Normalize

    :param config: CFG
    :return: (train transform, test transform)
    """
    train_transform = Compose([
        ChannelShuffle(p=0.5),
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


def transform_v5(config):
    """ GridDropout, VFlip, Hflip, Cutout, Normalize

    :param config: CFG
    :return: (train transform, test transform)
    """
    train_transform = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        GridDropout(random_offset=True, p=0.5),
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


def transform_v6(config):
    """ GridDropout * n, VFlip, Hflip, Cutout, Normalize

    :param config: CFG
    :return: (train transform, test transform)
    """

    grid_dropout = OneOf([
        GridDropout(holes_number_x=4, holes_number_y=4, random_offset=True),
        GridDropout(holes_number_x=6, holes_number_y=6, random_offset=True),
        GridDropout(holes_number_x=8, holes_number_y=8, random_offset=True),
        GridDropout(holes_number_x=10, holes_number_y=10, random_offset=True),
        GridDropout(random_offset=True),
    ])

    train_transform = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        grid_dropout,
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


def transform_v7(config):
    """ GridShuffle * n, GridDropout * n, VFlip, Hflip, Cutout, Normalize

    :param config: CFG
    :return: (train transform, test transform)
    """

    grid_shuffle = OneOf([
        RandomGridShuffle((2, 2)),
        RandomGridShuffle((4, 4)),
        RandomGridShuffle((6, 6)),
    ])

    grid_dropout = OneOf([
        GridDropout(holes_number_x=6, holes_number_y=6, random_offset=True),
        GridDropout(holes_number_x=8, holes_number_y=8, random_offset=True),
        GridDropout(holes_number_x=10, holes_number_y=10, random_offset=True),
    ])

    train_transform = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        OneOf([grid_shuffle, grid_dropout], p=0.66),
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


def get_transform(config):
    try:
        name = f"transform_v{config.transform_version}"
        f = globals().get(name)
        print(f"... Transform Info - {name}")
        return f(config)

    except TypeError:
        raise NotImplementedError("try another transform version ...")