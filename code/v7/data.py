import os
import pandas as pd
import numpy as np

import cv2

import torch
from torch.utils.data import Dataset


def load_data(config, train_sample_size=5000, valid_sample_size=1000):
    path = os.path.join(config.root_path, "dataset_info")
    # stegano algorithm
    algorithm = ['JMiPOD', 'JUNIWARD', 'UERD']
    # quality factor
    qfs = [75, 90, 95]

    # train dataframe
    train_filenames = []

    for i, algo in enumerate(algorithm):
        for j, qf in enumerate(qfs):
            names = pd.read_csv(os.path.join(path, f"train_dataset_info_qf{qf}.csv"))['image_filename']
            names = names.sample(train_sample_size).values
            names = [os.path.join(config.root_path, algo, name) for name in names]
            train_filenames.extend(names)

    train_df = pd.DataFrame({"ImageFileName": train_filenames})

    # valid dataframe
    valid_filenames = []

    for i, algo in enumerate(algorithm):
        for j, qf in enumerate(qfs):
            names = pd.read_csv(os.path.join(path, f"val_dataset_info_qf{qf}.csv"))['image_filename']
            names = names.sample(valid_sample_size).values
            names = [os.path.join(config.root_path, algo, name) for name in names]
            valid_filenames.extend(names)

    valid_df = pd.DataFrame({"ImageFileName": valid_filenames})

    def func(fn):
        fn = fn.split("/")
        fn[2] = "Cover"
        return "/".join(fn)

    train_df['CoverFileName'] = train_df['ImageFileName'].apply(func)
    valid_df['CoverFileName'] = valid_df['ImageFileName'].apply(func)

    print(f"... Shape Info - Train: {train_df.shape}, Valid: {valid_df.shape}")

    return train_df, valid_df


class Alaska2Dataset(Dataset):

    def __init__(self, config, df, augmentations=None):
        self.config = config
        self.data = df.values
        self.augment = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn_stego, fn_cover = self.data[idx]
        im_stego = cv2.imread(fn_stego)[:, :, ::-1]
        im_cover = cv2.imread(fn_cover)[:, :, ::-1]

        # Apply transformations
        if self.augment:
            im_stego = self.augment(image=im_stego)['image']
            im_cover = self.augment(image=im_cover)['image']

        return im_stego, im_cover

