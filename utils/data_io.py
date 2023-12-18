import os
import pickle
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

from utils.config_dataset import *


selected_pharma_file = os.path.join(path_processed, 'selected_pharma.p')
selected_physio_file = os.path.join(path_processed, 'selected_physio.csv')
selected_pharma = pickle.load(open(selected_pharma_file, 'rb'))
selected_physio = pd.read_csv(selected_physio_file)


def get_processed_path(file, processed_path=path_processed):
    return os.path.join(processed_path, file)

def pickle_load(path):
    return pickle.load(open(path, 'rb'))

def pickle_dump(data, path):
    pickle.dump(data, open(path, 'wb'))
    return 1


def read_patient_data(apache, pid, norm=True, processed_path=path_processed):
    if norm:
        file_path = get_processed_path(
            os.path.join('data_per_patient_resample2min_normalized',
                         apache.replace(' ', ''),
                         f'{pid}.csv'),
            processed_path
        )
    else:
        file_path = get_processed_path(
            os.path.join('data_per_patient_resample2min',
                         apache.replace(' ', ''),
                         f'{pid}.csv'),
            processed_path
        )
    return pd.read_csv(file_path, header=[0, 1], index_col=[0])


class HIRIDDataset(Dataset):
    """HIRID dataset."""

    def __init__(self, data_path, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pickle_load(data_path)
        self.transform = transform
        self.sample_index = self.data.index.droplevel(2).unique()

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_data = self.data.loc[self.data.index.droplevel(2)==self.sample_index[idx]]
        num_data = sample_data.loc[:, ('physio_num', slice(None))]
        cat_data = sample_data.loc[:, ('physio_cat', slice(None))]
        pharma_data = sample_data.loc[:, ('pharma', slice(None))]
        mask_missing_num = self.__get_missing_data_mask(num_data, 'num')
        mask_missing_cat = self.__get_missing_data_mask(cat_data, 'cat')
        mask_pharma = self.__get_pharma_mask(pharma_data)
        sample = {
            'num': num_data.replace(np.nan, -1),
            'cat': cat_data,
            'pharma': pharma_data,
            'mask_missing_num': mask_missing_num,
            'mask_missing_cat': mask_missing_cat,
            'mask_pharma': mask_pharma
        }

        # if self.transform:
        #     sample = self.transform(sample)

        return sample

    def __get_missing_data_mask(self, data, type='num'):
        if type == 'num':
            mask_missing = pd.isnull(data)
        elif type == 'cat':
            ind = data.columns
            mask_missing = pd.DataFrame()
            for uid in COL_PHYSIO_CAT:
                data_ = data.loc[:, ([x for x in ind if x[1].startswith(str(uid))])].astype(int)
                data_ = data_.astype(int)
                mask_ = (data_ == 0).all(1)
                mask_ = pd.concat([mask_.to_frame()] * len(data_.columns), axis=1)
                mask_.columns = data_.columns
                mask_missing = pd.concat((mask_missing, mask_), axis=1)
        return mask_missing

    def __get_pharma_mask(self, data):
        mask_pharma = (~pd.isnull(data.replace(0, np.nan))).any(1)
        #         ts_pharma = sample_pharma.index.get_level_values(2)[mask_pharma].to_numpy()
        return mask_pharma

