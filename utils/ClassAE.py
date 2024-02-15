import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


def mae_loss(x_, x):
    v = x[:, :, 2:]
    v_ = x_
    mask_valid = v!=-1
    # mask = x[:, :, 0]
    loss = torch.mean(torch.abs(v[mask_valid] - v_[mask_valid]))
    return loss

def mse_loss(x_, x):
    v = x[:, :, 2:]
    v_ = x_
    mask_valid = v!=-1
    # mask = x[:, :, 0]
    loss = torch.mean((v[mask_valid] - v_[mask_valid])**2)
    return loss


class LSTMEncoder(nn.Module):
    def __init__(self, n_feat=12, n_emb=128, n_layer=2, dropout=0):
        super().__init__()
        self.n_feat = n_feat
        self.n_emb = n_emb
        self.n_layer = n_layer
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=self.n_feat,
            hidden_size=self.n_emb,
            num_layers=self.n_layer,
            dropout=self.dropout,
            batch_first=True
        )

    def forward(self, x):
        out, hn = self.lstm(x.float())

        return out, hn



class LSTMDecoder(nn.Module):
    def __init__(self, config, seq_len=181, n_feat=12, n_emb=128, n_layer=2, dropout=0):
        super().__init__()
        self.config = config
        self.seq_len = seq_len
        self.n_feat = n_feat - 2
        self.n_emb = n_emb
        self.n_layer = n_layer
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=self.n_emb,
            hidden_size=self.n_emb,
            num_layers=self.n_layer,
            dropout=self.dropout,
            batch_first=True
        )
        self.out = nn.LazyLinear(out_features=self.n_feat)

    def forward(self, enc_hn):
        x_reconstruct = []
        hn = enc_hn
        N = enc_hn[0].shape[1]
        dec_in = torch.zeros(N, self.n_layer, self.n_emb).to(self.config['device'])
        # print(f'111 hidden: {hn[0].shape}, input: {dec_in.shape}')
        for _ in range(self.seq_len):
            # print(f'222 hidden: {hn[0].shape}, input: {dec_in.shape}')
            oi, hi = self.lstm(dec_in, hn)
            out = self.out(oi.reshape(oi.shape[0], -1))
            # print(f'FC output: {out.shape}, input: {oi.shape}')
            out = F.sigmoid(out)            # check if it worsens/improves the performance
            out = out[:, None, :]
            # print(f'FC output: {out.shape}')
            x_reconstruct.append(out)
            # print(f'LSTM OUTPUT {oi.shape}')
            # dec_in = oi.permute(1, 0, 2)
            dec_in = oi
            hn = hi
            # print(f'333 hidden: {hn[0].shape}, input: {dec_in.shape}')

        # print(f'X_reconstruction: {len(x_reconstruct)}')
        x_reconstruct = torch.cat(x_reconstruct, dim=1)
        # print(f'X_reconstruction: {x_reconstruct.shape}')
        return x_reconstruct



class LSTM_AE(LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.seq_len = config['seq_len']
        self.n_feat = config['n_feat']
        self.n_emb = config['n_emb']
        self.n_layer = config['n_layer']
        self.dropout = config['dropout']

        self.encoder = LSTMEncoder(n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)
        self.decoder = LSTMDecoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)

        self.METRICS = {
            'mse': mse_loss,
        }

    def forward(self, x):
        _, enc_hn = self.encoder(x)
        x_ = self.decoder(enc_hn)

        return x_

    def configure_optimizers(self):
        adam = optim.Adam(self.parameters(), lr=self.config['lr'])
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, mode='min', factor=.5, patience=5)
        return [adam], {"scheduler": lr_scheduler, "monitor": "train_loss"}
        # return adam

    def training_epoch_end(self, outputs):
        lr_sch = self.lr_schedulers()
        lr_sch.step(self.trainer.callback_metrics["train_loss"])

    def training_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x)
        # loss = F.l1_loss(x_, x)
        loss = mae_loss(x_, x)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](x_, x), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x)
        # loss = F.l1_loss(x_, x)
        loss = mae_loss(x_, x)

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](x_, x)
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x)
        # loss = F.l1_loss(x_, x)
        loss = mae_loss(x_, x)

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](x_, x)
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

