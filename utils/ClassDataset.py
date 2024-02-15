from torch.utils.data import Dataset

class CusDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data_ = self.data[idx]
        # vitals = data_[:, 2:]
        med_mask = data_[:, 0]
        med = data_[90, 1]

        return {
            'data': data_,
            'dosage': med,
            'med_mask': med_mask
        }