import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger



def mae_loss(x_, x):
    mask_valid = x!=-1
    # mask = x[:, :, 0]
    loss = torch.mean(torch.abs(x[mask_valid] - x_[mask_valid]))
    return loss

def mse_loss(x_, x):
    mask_valid = x!=-1
    # mask = x[:, :, 0]
    loss = torch.mean((x[mask_valid] - x_[mask_valid])**2)
    return loss

def index_map(n):
    index_map = {i: (i // n, i % n) for i in range(n**2)}
    index_map_invert = {j: i for i, j in index_map.items()}

    return index_map, index_map_invert


class BaseModelSOM(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # input params
        self.batchsize = config['batchsize']
        self.seq_len = config['seq_len']
        self.n_feat = config['n_feat']
        self.input_len = int(self.seq_len // 2)
        self.pred_len = int(self.seq_len // 2)
        # model params
        self.encoder_type = config['encoder_type']
        self.n_emb = config['n_emb']
        self.n_layer = config['n_layer']
        self.dropout = config['dropout']
        # SOM params
        self.som_size = config['som_size']
        self.r_neighbor = config['r_neighbor']
        self.emb_dict = torch.rand(self.batchsize, self.n_emb)
        self.som_map, self.som_map_invert = index_map(self.som_size)

        self.METRICS = {
            'mse': mse_loss,
        }

    def configure_optimizers(self):
        adam = optim.Adam(self.parameters(), lr=self.config['lr'])
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, mode='min', factor=.5, patience=5)
        return [adam], {"scheduler": lr_scheduler, "monitor": "train_loss"}
        # return adam

    def training_epoch_end(self, outputs):
        lr_sch = self.lr_schedulers()
        lr_sch.step(self.trainer.callback_metrics["train_loss"])

    def neighborhood(self, index):
        x, y = self.som_map(index)
        r = self.r_neighbor
        n = self.som_size

        neighbors = []
        for i in range(max(0, x-r), min(n, x+r+1)):
            for j in range(max(0, y-r), min(n, y+r+1)):
                dist = np.linalg.norm((x-i, y-j))
                if dist <= r:
                    neighbors += (i, j)
        return neighbors

    def init_encoder(self):
        if self.encoder_type == 'CNN':
            self.encoder_ae = nn.Conv1d(in_channels=7, out_channels=56, kernel_size=6, stride=3, groups=7)
            self.encoder_physio = nn.Conv1d(in_channels=7, out_channels=56, kernel_size=6, stride=3, groups=7)
            self.encoder_med = nn.Conv1d(in_channels=7, out_channels=35, kernel_size=6, stride=3, groups=7)
        elif self.encoder_type == 'LSTM':
            pass
        elif self.encoder_type == 'ATT':
            pass
        else:
            pass



class SOM_CLS(BaseModelSOM):
    # Classifier for medication type
    def __init__(self):
        self.init_encoder()
        self.decoder = nn.Conv1d()
        pass


class SOM_PRED(BaseModelSOM):
    # Predictor for physio trend when taking medication
    def __init__(self):
        pass


class SOM_MTL(BaseModelSOM):
    # Multi-task learning for both CLS and PRED
    def __init__(self):
        pass