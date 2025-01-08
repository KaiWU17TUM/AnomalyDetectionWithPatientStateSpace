import pickle
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from utils.config_dataset import MED_BENCHMARK, PHYSIO_BENCHMARK, PHYSIO_BENCHMARK_ALL, PHYSIO_BENCHMARK_CAT, NUM_CAT_PHYSIO_BENCHMARK, APACHE_BENCHMARK_MERGE_INDEX
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib.pyplot as plt

# class CusDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
#
#     def __len__(self):
#         return self.data.shape[0]
#
#     def __getitem__(self, idx):
#         data_ = self.data[idx]
#         # vitals = data_[:, 2:]
#         med_mask = data_[:, 0]
#         med = data_[:90, 1:10]
#
#         return {
#             'data': data_,
#             'dosage': med,
#             'med_mask': med_mask
#         }
#
# class ClassifierDataset(Dataset):
#     def __init__(self, data, labels, target='survived'):
#         self.X, self.X_rec, self.X_enc = data
#         self.info = labels
#         self.target = target
#         self.label = labels[target]
#
#     def __len__(self):
#         return self.X.shape[0]
#
#     def get_labels(self):
#         return self.label
#
#     def __getitem__(self, idx):
#         x = self.X[idx]
#         x_rec = self.X_rec[idx]
#         x_enc = self.X_enc[idx]
#
#         age = self.info['age'][idx]
#         sex = self.info['sex'][idx]
#         sex = F.one_hot(torch.Tensor([sex]).long(), num_classes=2)
#         apache = self.info['apache'][idx]
#         apache = F.one_hot(torch.Tensor([apache]).long(), num_classes=3)
#         los = self.info['los'][idx]
#         loglos = self.info['loglos'][idx]
#         survived = self.info['survived'][idx]
#
#         return {
#             'data': (x, x_rec, x_enc),
#             'info': {
#                 'age': age,
#                 'sex': sex,
#                 'apache':apache,
#                 'los': los,
#                 'loglos': loglos,
#                 'survived': survived,
#             },
#             'label': self.label[idx],
#         }
#
# class RegressionDataset(Dataset):
#     def __init__(self, data, labels):
#         self.X, self.X_rec, self.X_enc = data
#         self.info = labels
#         self.label = labels['loglos']
#
#     def __len__(self):
#         return self.X.shape[0]
#
#     def get_labels(self):
#         return self.label
#
#     def __getitem__(self, idx):
#         x = self.X[idx]
#         x_rec = self.X_rec[idx]
#         x_enc = self.X_enc[idx]
#
#         age = self.info['age'][idx]
#         sex = self.info['sex'][idx]
#         sex = F.one_hot(torch.Tensor([sex]).long(), num_classes=2)
#         apache = self.info['apache'][idx]
#         apache = F.one_hot(torch.Tensor([apache]).long(), num_classes=3)
#         los = self.info['los'][idx]
#         survived = self.info['survived'][idx]
#
#         return {
#             'data': (x, x_rec, x_enc),
#             'info': {
#                 'age': age,
#                 'sex': sex,
#                 'apache':apache,
#                 'los': los,
#                 'survived': survived,
#             },
#             'label': self.label[idx],
#         }
#
# class BenchmarkAEDataset(Dataset):
#     def __init__(self, data, freq='6h', norm=True, selected_physio=True):
#         self.data = data
#         self.pids = data['patientid'].unique()
#         self.col_id = 'sampleid_' + freq
#         self.sample_ids = list(data[self.col_id].unique())
#         self.sample_ids.remove(-1)
#         self.norm = norm
#         self.selected_physio = selected_physio
#         self.norm_params = pd.read_csv('processed-benchmark/patient_data_statistics.csv', header=[0], index_col=[0])
#
#         # self.INFO = ['age', 'height', 'sex', 'APACHE MERGED']
#
#     def __len__(self):
#         return len(self.sample_ids)
#
#     def __getitem__(self, idx):
#         sid = self.sample_ids[idx]
#         sample = self.data[self.data[self.col_id] == sid].sort_values('datetime')
#         try:
#             assert len(sample['patientid'].unique()) == 1
#         except:
#             print(111)
#
#         med = sample[MED_BENCHMARK]
#         if self.selected_physio:
#             data_num = sample[PHYSIO_BENCHMARK]
#             data_cat = None
#         else:
#             data_num = sample[[col for col in PHYSIO_BENCHMARK_ALL if col not in PHYSIO_BENCHMARK_CAT]]
#             data_cat = sample[PHYSIO_BENCHMARK_CAT]
#
#         age = sample['age'].iloc[0]
#         height = sample['height'].iloc[0]
#         sex = sample['sex'].iloc[0]
#         sex = 1 if sex == 'M' else 0
#         sex = F.one_hot(torch.Tensor([sex]).long(), num_classes=2)
#         apache = sample['APACHE MERGED'].iloc[0]
#         apache = APACHE_BENCHMARK_MERGE_INDEX[apache]
#         apache = F.one_hot(torch.Tensor([apache]).long(), num_classes=15)
#
#         data_cat_onehot = torch.Tensor()
#         if data_cat:
#             # TODO: NAN as row of zeros
#             for col in data_cat.columns:
#                 ohe = F.one_hot(torch.tensor(data_cat[col]-1).long(), num_classes=NUM_CAT_PHYSIO_BENCHMARK[col])
#                 data_cat_onehot = torch.concat((data_cat_onehot, ohe), dim=1)
#
#         if self.norm:
#             age = (age - self.norm_params['age'].loc['min']) / (self.norm_params['age'].loc['max'] - self.norm_params['age'].loc['min'])
#             height = (height - self.norm_params['height'].loc['min']) / (self.norm_params['height'].loc['max'] - self.norm_params['height'].loc['min'])
#             med = self.norm_numeric_data(med)
#             data_num = self.norm_numeric_data(data_num)
#
#             age = .5 if age == np.nan else age
#             height = .5 if height == np.nan else height
#             data_num.fillna(-1, inplace=True)
#             # data_num.fillna(0.5, inplace=True)
#
#
#         med.fillna(0, inplace=True)
#
#         return {
#             'info': {
#                 'age': age,
#                 'height': height,
#                 'sex': sex,
#                 'apache': apache,
#             },
#             'data': {
#                 'med': med.to_numpy(),
#                 'num': data_num.to_numpy(),
#                 'cat': data_cat_onehot,
#             }
#         }
#
#     def norm_numeric_data(self, data_num):
#         COLS = data_num.columns
#         data_num = (data_num - self.norm_params[COLS].loc['0.1%']) / (self.norm_params[COLS].loc['99.9%'] - self.norm_params[COLS].loc['0.1%'])
#         data_num[data_num > 1] = 1
#
#         return data_num



class MergedDataset(Dataset):
    def __init__(self, base_path, sample_dict, df_info, type='vaso', norm=True, smooth=True, interpolate=True,
                 n_step=3, n_step_med=15,
                 selected_physio=None, selected_med=None):
        self.base_path = base_path
        self.sample_dict = sample_dict
        self.df_info = df_info
        self.type = type
        self.norm = norm
        self.smooth = smooth
        self.interpolate = interpolate
        self.n_step = n_step
        self.n_step_med = n_step_med
        self.selected_physio = selected_physio
        self.selected_med = selected_med
        if norm:
            self.norm_params = pickle.load(open(os.path.join(self.base_path, 'norm_params_vasopressor.p'), 'rb'))
            self.norm_params_info = pickle.load(open(os.path.join(self.base_path, 'norm_params_info.p'), 'rb'))
        # self.INFO = ['age', 'height', 'sex', 'APACHE MERGED']

    def __len__(self):
        return len(self.sample_dict)

    def __getitem__(self, idx):

        sampleid = list(self.sample_dict.keys())[idx]
        if self.type == 'vaso':
            med_name, pid, t_start, t_end = self.sample_dict[sampleid]

            med_label = self.selected_med.index(med_name)
            if med_label == 0 or med_label == 'norepinephrine':
                med_label = np.array([1, 0, 0])
            elif med_label == 1 or med_label == 'epinephrine':
                med_label = np.array([0, 1, 0])
            elif med_label == 2 or med_label == 'dobutamine':
                med_label = np.array([0, 0, 1])
            else:
                raise ValueError(f"MED_LABEL: {med_label}")
        elif self.type == 'control':
            med_label = np.array([0,0,0])
            pid, t_start, t_end = self.sample_dict[sampleid]
        # patient information
        info = self.df_info[self.df_info['patientid']==pid]
        df = pickle.load(open(os.path.join(os.path.join(self.base_path, 'merged_data_per_pat'), f"{pid}.p"), 'rb'))
        # time-series vital sign
        sample = df.loc[(df.index>=t_start) & (df.index<t_end)]
        data = sample[self.selected_physio]
        data_mask = ~pd.isnull(data)
        # med infusion
        med = sample[MED_BENCHMARK]
        med[pd.isnull(med)] = 0
        resp_failure = sample['resp_failure_status'].values
        circ_failure = sample['circ_failure_status'].values
        los = sample['LOS'].values

        age = info['age'].item()
        height = info['height'].item()
        sex = info['sex'].item()
        sex = 1 if sex == 'M' else 0
        sex = F.one_hot(torch.Tensor([sex]).long(), num_classes=2).flatten()
        apache = info['APACHE MERGED'].item()
        apache = APACHE_BENCHMARK_MERGE_INDEX[apache]
        apache = F.one_hot(torch.Tensor([apache]).long(), num_classes=15).flatten()

        if self.norm:
            age = (age - self.norm_params_info['age'].loc['min']) / (self.norm_params_info['age'].loc['max'] - self.norm_params_info['age'].loc['min'])
            height = (height - self.norm_params_info['height'].loc['min']) / (self.norm_params_info['height'].loc['max'] - self.norm_params_info['height'].loc['min'])
            data = self.norm_numeric_data(data)
            med = self.norm_numeric_data(med)

            age = .5 if np.isnan(age) else age
            height = .5 if np.isnan(height) else height

        if self.interpolate:
            data.interpolate(method='linear', limit_area='inside', inplace=True)
            data.interpolate('ffill', inplace=True)
            data.interpolate('bfill', inplace=True)
            data.fillna(.5, inplace=True)

        if self.smooth:
            for col in self.selected_physio:
                data_ = data.reset_index()[col]
                # data_nan = pd.isnull(data_).sum()
                # data_ = data_[~pd.isnull(data_)]
                x = data_.index.to_numpy()
                y = data_.values
                smoothed = lowess(exog=x, endog=y, frac=0.05, missing='drop', is_sorted=True)
                # smoothed_nan = np.isnan(smoothed).sum()
                # plt.plot(data[col].values)
                # plt.plot(smoothed[:,0], smoothed[:,1], linestyle='--')
                # plt.title(col)
                # plt.show()
                data[col].iloc[smoothed[:,0].astype(int)] = smoothed[:,1]

        # data_regression = data.rolling(self.n_step).mean().shift(self.n_step - 1)
        data_regression = data.rolling(self.n_step, center=True).mean().shift(-1)

        med_acc = med.rolling(self.n_step_med).sum()
        med_acc[pd.isnull(med_acc)] = 0
        dosage_trend = med[med_name].diff()
        dosage_trend[pd.isnull(dosage_trend)] = 0
        dosage_trend_bool = dosage_trend.copy()
        dosage_trend_bool[dosage_trend_bool > 0] = 1
        dosage_trend_bool[dosage_trend_bool < 0] = 2
        dosage_trend_bool = F.one_hot(torch.Tensor(dosage_trend_bool).long(), num_classes=3)
        if dosage_trend_bool.shape != (180,3):
            print(dosage_trend_bool.shape)
        # # print(f"DOSAGE BOOL - {dosage_trend_bool.shape}")


        return {
            'info': {
                'age': torch.Tensor([age]),
                'height': torch.Tensor([height]),
                'sex': sex,
                'apache': apache,
            },
            'data': data.to_numpy(),
            'data_mask': data_mask.to_numpy(),
            'data_regression': data_regression.to_numpy(),
            'med': med.to_numpy(),
            'med_label': med_label,
            'med_acc': med_acc.to_numpy(),
            'dosage_trend': dosage_trend.to_numpy(),
            'dosage_trend_bool': dosage_trend_bool,
            'resp_failure': resp_failure,
            'circ_failure': circ_failure,
            'los': los,
        }

    def norm_numeric_data(self, data_num):
        COLS = data_num.columns
        for col in COLS:
            data_num.loc[data_num[col] > self.norm_params[col].loc['99.9%'], col] = self.norm_params[col].loc['99.9%']
            data_num.loc[data_num[col] < self.norm_params[col].loc['0.1%'], col] = self.norm_params[col].loc['0.1%']
        data_num = (data_num - self.norm_params[COLS].loc['0.1%']) / (self.norm_params[COLS].loc['99.9%'] - self.norm_params[COLS].loc['0.1%'])

        return data_num






if __name__ == '__main__':
    data_path = 'processed-merge/'
    pid_valid = pickle.load(open(os.path.join(data_path, 'pid_valid_00.p'), 'rb'))
    patient_info = pickle.load(open(os.path.join(data_path, 'patient_info.p'), 'rb'))
    sample_dict = pickle.load(open(os.path.join(data_path, 'sample_dict_vasopressor.p'), 'rb'))
    selected_physio = ['HR', 'RR', 'SpO2', 'ABPd', 'ABPm', 'ABPs', 'ZVD']
    selected_med = ['norepinephrine', 'epinephrine', 'dobutamine']

    dataset = MergedDataset(basepath=data_path, sample_dict=sample_dict, df_info=patient_info, norm=True, selected_physio=selected_physio, selected_med=selected_med)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for sample in loader:
        info = sample['info']
        data = sample['data'].numpy()
        med = sample['med']
        print(info['age'], info['height'])

