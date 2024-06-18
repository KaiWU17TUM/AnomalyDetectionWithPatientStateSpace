import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils.config_dataset import MED_BENCHMARK, PHYSIO_BENCHMARK, PHYSIO_BENCHMARK_ALL, PHYSIO_BENCHMARK_CAT, NUM_CAT_PHYSIO_BENCHMARK, APACHE_BENCHMARK_MERGE_INDEX

class CusDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data_ = self.data[idx]
        # vitals = data_[:, 2:]
        med_mask = data_[:, 0]
        med = data_[:90, 1:10]

        return {
            'data': data_,
            'dosage': med,
            'med_mask': med_mask
        }

class ClassifierDataset(Dataset):
    def __init__(self, data, labels, target='survived'):
        self.X, self.X_rec, self.X_enc = data
        self.info = labels
        self.target = target
        self.label = labels[target]

    def __len__(self):
        return self.X.shape[0]

    def get_labels(self):
        return self.label

    def __getitem__(self, idx):
        x = self.X[idx]
        x_rec = self.X_rec[idx]
        x_enc = self.X_enc[idx]

        age = self.info['age'][idx]
        sex = self.info['sex'][idx]
        sex = F.one_hot(torch.Tensor([sex]).long(), num_classes=2)
        apache = self.info['apache'][idx]
        apache = F.one_hot(torch.Tensor([apache]).long(), num_classes=3)
        los = self.info['los'][idx]
        loglos = self.info['loglos'][idx]
        survived = self.info['survived'][idx]

        return {
            'data': (x, x_rec, x_enc),
            'info': {
                'age': age,
                'sex': sex,
                'apache':apache,
                'los': los,
                'loglos': loglos,
                'survived': survived,
            },
            'label': self.label[idx],
        }

class RegressionDataset(Dataset):
    def __init__(self, data, labels):
        self.X, self.X_rec, self.X_enc = data
        self.info = labels
        self.label = labels['loglos']

    def __len__(self):
        return self.X.shape[0]

    def get_labels(self):
        return self.label

    def __getitem__(self, idx):
        x = self.X[idx]
        x_rec = self.X_rec[idx]
        x_enc = self.X_enc[idx]

        age = self.info['age'][idx]
        sex = self.info['sex'][idx]
        sex = F.one_hot(torch.Tensor([sex]).long(), num_classes=2)
        apache = self.info['apache'][idx]
        apache = F.one_hot(torch.Tensor([apache]).long(), num_classes=3)
        los = self.info['los'][idx]
        survived = self.info['survived'][idx]

        return {
            'data': (x, x_rec, x_enc),
            'info': {
                'age': age,
                'sex': sex,
                'apache':apache,
                'los': los,
                'survived': survived,
            },
            'label': self.label[idx],
        }



class BenchmarkAEDataset(Dataset):
    def __init__(self, data, freq='6h', norm=True, selected_physio=True):
        self.data = data
        self.pids = data['patientid'].unique()
        self.col_id = 'sampleid_' + freq
        self.sample_ids = list(data[self.col_id].unique())
        self.sample_ids.remove(-1)
        self.norm = norm
        self.selected_physio = selected_physio
        self.norm_params = pd.read_csv('processed-benchmark/patient_data_statistics.csv', header=[0], index_col=[0])

        # self.INFO = ['age', 'height', 'sex', 'APACHE MERGED']

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        sample = self.data[self.data[self.col_id] == sid].sort_values('datetime')
        try:
            assert len(sample['patientid'].unique()) == 1
        except:
            print(111)

        med = sample[MED_BENCHMARK]
        if self.selected_physio:
            data_num = sample[PHYSIO_BENCHMARK]
            data_cat = None
        else:
            data_num = sample[[col for col in PHYSIO_BENCHMARK_ALL if col not in PHYSIO_BENCHMARK_CAT]]
            data_cat = sample[PHYSIO_BENCHMARK_CAT]

        age = sample['age'].iloc[0]
        height = sample['height'].iloc[0]
        sex = sample['sex'].iloc[0]
        sex = 1 if sex == 'M' else 0
        sex = F.one_hot(torch.Tensor([sex]).long(), num_classes=2)
        apache = sample['APACHE MERGED'].iloc[0]
        apache = APACHE_BENCHMARK_MERGE_INDEX[apache]
        apache = F.one_hot(torch.Tensor([apache]).long(), num_classes=15)

        data_cat_onehot = torch.Tensor()
        if data_cat:
            # TODO: NAN as row of zeros
            for col in data_cat.columns:
                ohe = F.one_hot(torch.tensor(data_cat[col]-1).long(), num_classes=NUM_CAT_PHYSIO_BENCHMARK[col])
                data_cat_onehot = torch.concat((data_cat_onehot, ohe), dim=1)

        if self.norm:
            age = (age - self.norm_params['age'].loc['min']) / (self.norm_params['age'].loc['max'] - self.norm_params['age'].loc['min'])
            height = (height - self.norm_params['height'].loc['min']) / (self.norm_params['height'].loc['max'] - self.norm_params['height'].loc['min'])
            med = self.norm_numeric_data(med)
            data_num = self.norm_numeric_data(data_num)

            age = .5 if age == np.nan else age
            height = .5 if height == np.nan else height
            data_num.fillna(-1, inplace=True)
            # data_num.fillna(0.5, inplace=True)


        med.fillna(0, inplace=True)

        return {
            'info': {
                'age': age,
                'height': height,
                'sex': sex,
                'apache': apache,
            },
            'data': {
                'med': med.to_numpy(),
                'num': data_num.to_numpy(),
                'cat': data_cat_onehot,
            }
        }

    def norm_numeric_data(self, data_num):
        COLS = data_num.columns
        data_num = (data_num - self.norm_params[COLS].loc['0.1%']) / (self.norm_params[COLS].loc['99.9%'] - self.norm_params[COLS].loc['0.1%'])
        data_num[data_num > 1] = 1

        return data_num

