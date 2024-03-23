import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
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