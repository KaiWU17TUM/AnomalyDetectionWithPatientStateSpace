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
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

from utils.config_dataset import *
from utils.ClassDataset import MergedDataset


#
# selected_pharma_file = os.path.join(path_processed, 'selected_pharma.p')
# selected_physio_file = os.path.join(path_processed, 'selected_physio.csv')
# selected_pharma = pickle.load(open(selected_pharma_file, 'rb'))
# selected_physio = pd.read_csv(selected_physio_file)
#
#
# def get_processed_path(file, processed_path=path_processed):
#     return os.path.join(processed_path, file)

def pickle_load(path):
    return pickle.load(open(path, 'rb'))

def pickle_dump(data, path):
    pickle.dump(data, open(path, 'wb'))
    return 1

def load_train_test_dataset(batchsize, RANDOMSEED=2024, norm=True, smooth=True):
    # load data
    print("Loading dataset")
    data_path = 'processed-merge/'
    pid_valid = pickle.load(open(os.path.join(data_path, 'pid_valid_00.p'), 'rb'))
    patient_info = pickle.load(open(os.path.join(data_path, 'patient_info.p'), 'rb'))
    sample_dict = pickle.load(open(os.path.join(data_path, 'sample_dict_vasopressor.p'), 'rb'))
    norm_params = pickle.load(open('processed-merge/norm_params_vasopressor.p', 'rb'))
    norm_params_info = pickle.load(open('processed-merge/norm_params_info_vasopressor.p', 'rb'))


    med_labels = np.array([sample_dict[i][0] for i in range(len(sample_dict))])
    selected_physio = ['HR', 'RR', 'SpO2', 'ABPd', 'ABPm', 'ABPs', 'ZVD']
    selected_med = ['norepinephrine', 'epinephrine', 'dobutamine']
    sampleid_train, sampleid_test = train_test_split(list(sample_dict.keys()),
                                                 test_size=0.2,
                                                 random_state=RANDOMSEED,
                                                 stratify=med_labels)
    sampleid_tr, sampleid_val = train_test_split(sampleid_train, test_size=0.2, random_state=RANDOMSEED,
                                                 stratify=med_labels[sampleid_train])

    dataset_train = MergedDataset(
        sample_dict={item[0]: item[1] for item in sample_dict.items() if item[0] in sampleid_tr},
        df_info=patient_info, norm=norm, smooth=smooth, selected_physio=selected_physio, selected_med=selected_med)
    dataset_val = MergedDataset(
        sample_dict={item[0]: item[1] for item in sample_dict.items() if item[0] in sampleid_val},
        df_info=patient_info, norm=norm, smooth=smooth, selected_physio=selected_physio, selected_med=selected_med)
    dataset_test = MergedDataset(
        sample_dict={item[0]: item[1] for item in sample_dict.items() if item[0] in sampleid_test},
        df_info=patient_info, norm=norm, smooth=smooth, selected_physio=selected_physio, selected_med=selected_med)
    loader_train = DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=batchsize)
    loader_val = DataLoader(dataset_val, batch_size=batchsize, shuffle=False, num_workers=batchsize)
    loader_test = DataLoader(dataset_test, batch_size=batchsize, shuffle=False, num_workers=batchsize)

    return {
        'pid_valid': pid_valid,
        'norm_params': norm_params,
        'norm_params_info': norm_params_info,
        'patient_info': patient_info,
        'dataset_train': dataset_train,
        'dataset_val': dataset_val,
        'dataset_test': dataset_test,
        'loader_train': loader_train,
        'loader_val': loader_val,
        'loader_test': loader_test,
    }





# def read_patient_data(apache, pid, norm=True, processed_path=path_processed):
#     if norm:
#         file_path = get_processed_path(
#             os.path.join('data_per_patient_resample2min_normalized',
#                          apache.replace(' ', ''),
#                          f'{pid}.csv'),
#             processed_path
#         )
#     else:
#         file_path = get_processed_path(
#             os.path.join('data_per_patient_resample2min',
#                          apache.replace(' ', ''),
#                          f'{pid}.csv'),
#             processed_path
#         )
#     return pd.read_csv(file_path, header=[0, 1], index_col=[0])




# class HIRIDDataset(Dataset):
#     """HIRID dataset."""
#
#     def __init__(self, data_path, transform=None):
#         """
#         Arguments:
#             csv_file (string): Path to the csv file with annotations.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.data = pickle_load(data_path)
#         self.transform = transform
#         self.sample_index = self.data.index.droplevel(2).unique()
#
#     def __len__(self):
#         return len(self.sample_index)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         sample_data = self.data.loc[self.data.index.droplevel(2)==self.sample_index[idx]]
#         num_data = sample_data.loc[:, ('physio_num', slice(None))]
#         cat_data = sample_data.loc[:, ('physio_cat', slice(None))]
#         pharma_data = sample_data.loc[:, ('pharma', slice(None))]
#         mask_missing_num = self.__get_missing_data_mask(num_data, 'num')
#         mask_missing_cat = self.__get_missing_data_mask(cat_data, 'cat')
#         mask_pharma = self.__get_pharma_mask(pharma_data)
#         sample = {
#             'num': num_data.replace(np.nan, -1),
#             'cat': cat_data,
#             'pharma': pharma_data,
#             'mask_missing_num': mask_missing_num,
#             'mask_missing_cat': mask_missing_cat,
#             'mask_pharma': mask_pharma
#         }
#
#         # if self.transform:
#         #     sample = self.transform(sample)
#
#         return sample
#
#     def __get_missing_data_mask(self, data, type='num'):
#         if type == 'num':
#             mask_missing = pd.isnull(data)
#         elif type == 'cat':
#             ind = data.columns
#             mask_missing = pd.DataFrame()
#             for uid in COL_PHYSIO_CAT:
#                 data_ = data.loc[:, ([x for x in ind if x[1].startswith(str(uid))])].astype(int)
#                 data_ = data_.astype(int)
#                 mask_ = (data_ == 0).all(1)
#                 mask_ = pd.concat([mask_.to_frame()] * len(data_.columns), axis=1)
#                 mask_.columns = data_.columns
#                 mask_missing = pd.concat((mask_missing, mask_), axis=1)
#         return mask_missing
#
#     def __get_pharma_mask(self, data):
#         mask_pharma = (~pd.isnull(data.replace(0, np.nan))).any(1)
#         #         ts_pharma = sample_pharma.index.get_level_values(2)[mask_pharma].to_numpy()
#         return mask_pharma


