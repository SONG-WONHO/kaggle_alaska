import os
import pandas as pd
import numpy as np

import cv2
import h5py
from scipy import io

import torch
from torch.utils.data import Dataset


def load_data(config, train_sample_size=5000, valid_sample_size=1000):
    path = os.path.join(config.root_path, "dataset_info")
    # stegano algorithm
    algorithm = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
    # quality factor
    qfs = [75, 90, 95]

    # target map
    mp = []
    for i in range(len(algorithm)):
        for j in range(len(qfs)):
            mp.append((i, j))

    mp = {v: i - 2 for i, v in enumerate(mp)}
    mp[(0, 0)] = 0
    mp[(0, 1)] = 0

    # train dataframe
    train_filenames = []
    train_labels = []
    train_qfs = []

    for i, algo in enumerate(algorithm):
        for j, qf in enumerate(qfs):
            names = pd.read_csv(os.path.join(path, f"train_dataset_info_qf{qf}.csv"))['image_filename']
            names = names.sample(train_sample_size).values
            names = [os.path.join(config.root_path, algo, name) for name in names]
            train_filenames.extend(names)
            train_labels += [(i, j)] * train_sample_size
            train_qfs += [j] * train_sample_size

    train_df = pd.DataFrame({"ImageFileName": train_filenames, "Label": train_labels, "qf": train_qfs})
    train_df['Label'] = train_df['Label'].map(mp)
    train_df['LabelBinary'] = (train_df['Label'] != 0) * 1

    # valid dataframe
    valid_filenames = []
    valid_labels = []
    valid_qfs = []

    for i, algo in enumerate(algorithm):
        for j, qf in enumerate(qfs):
            names = pd.read_csv(os.path.join(path, f"val_dataset_info_qf{qf}.csv"))['image_filename']
            names = names.sample(valid_sample_size).values
            names = [os.path.join(config.root_path, algo, name) for name in names]
            valid_filenames.extend(names)
            valid_labels += [(i, j)] * valid_sample_size
            valid_qfs += [j] * valid_sample_size

    valid_df = pd.DataFrame({"ImageFileName": valid_filenames, "Label": valid_labels, "qf": valid_qfs})
    valid_df['Label'] = valid_df['Label'].map(mp)
    valid_df['LabelBinary'] = (valid_df['Label'] != 0) * 1

    # test dataframe
    ss_df = pd.read_csv(os.path.join(config.root_path, "sample_submission.csv"))
    test_df = ("./input/Test/" + ss_df['Id']).to_frame()
    test_df['Label'] = np.nan
    test_df.columns = ['ImageFileName', 'Label']

    test_filenames = []
    test_qfs = []
    for j, qf in enumerate(qfs):
        names = pd.read_csv(os.path.join(path, f'test_dataset_info_qf{qf}.csv'))['image_filename'].values
        names = [os.path.join(config.root_path, "Test", name) for name in names]
        test_filenames.extend(names)
        test_qfs += [j] * len(names)

    test_df = test_df.merge(pd.DataFrame({"ImageFileName": test_filenames, "qf": test_qfs}),
                            on='ImageFileName',
                            how='left')

    test_df['LabelBinary'] = np.nan

    print(f"... Shape Info - Train: {train_df.shape}, Valid: {valid_df.shape}, Test: {test_df.shape}")

    return train_df, valid_df, test_df


def load_tabular(df):
    fns = []
    for fn in df['ImageFileName']:
        fn = fn.split("/")
        fn.insert(2, "DCTR")
        fn[-1] = fn[-1].replace(".jpg", ".mat")
        fn = "/".join(fn)
        fns.append(fn)

    fns_gfr = []
    for fn in df['ImageFileName']:
        fn = fn.split("/")
        fn.insert(2, "GFR")
        fn[-1] = fn[-1].replace(".jpg", ".mat")
        fn = "/".join(fn)
        fns_gfr.append(fn)

    df = pd.DataFrame({"DCTR": fns, "GFR": fns_gfr})
    return df


class Alaska2Dataset(Dataset):

    def __init__(self, config, df, tabular, augmentations=None):
        self.config = config
        self.data = df.values
        self.tabular = tabular.values
        self.augment = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn, label, qf, label_bin = self.data[idx]
        im = cv2.imread(fn)[:, :, ::-1]

        # Apply transformations
        if self.augment:
            im = self.augment(image=im)['image']

        dctr, gfr = self.tabular[idx]

        # load dctr feature
        dctr = h5py.File(dctr, 'r')
        dctr = np.transpose(np.array(dctr.get('DCTR_features')), (1, 0))

        # load gfr feature
        gfr = io.loadmat(gfr)['feat']

        return im, qf, label, label_bin, dctr, gfr
