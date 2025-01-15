import os

import numpy as np
import functools
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torch.autograd as autograd
import monotonicnetworks as lmn
from monotonicnetworks import GroupSort
from pytorch_tcn import TemporalConv1d, TemporalConvTranspose1d
from tslearn.metrics import SoftDTWLossPyTorch

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.data_io import pickle_load, pickle_dump
from utils.ClassMonoModel import AE_MED_PHYSIO
import matplotlib.pyplot as plt

med_dict = {
    0: [1, 0, 0],
    1: [0, 1, 0],
    2: [0, 0, 1],
}

class BaseCLF(LightningModule):
    def __init__(self, config):

        super().__init__()
        self.config = config
        # input params
        self.batchsize = config['batchsize']
        self.dropout = config['dropout']
        self.n_emb = config['n_emb']
        self.n_emb_info = config['n_emb_info']

        self.loss_clf = nn.BCEWithLogitsLoss()

        self.METRICS = {
            'roc': torchmetrics.AUROC(pos_label=1).to(self.config['device']),
            'acc': torchmetrics.Accuracy().to(device=self.config['device']),
            'recall': torchmetrics.Recall().to(device=self.config['device']),
            'precision': torchmetrics.Precision().to(device=self.config['device']),
            'f1': torchmetrics.F1().to(self.config['device']),
        }

    def configure_optimizers(self):
        adam = optim.Adam(self.parameters(), lr=self.config['lr'])
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, mode='min', factor=.5, patience=5)
        return [adam], {"scheduler": lr_scheduler, "monitor": "train_loss"}
        # return adam

    def training_epoch_end(self, outputs):
        lr_sch = self.lr_schedulers()
        lr_sch.step(self.trainer.callback_metrics["train_loss"])

class MED_CLF(BaseCLF):
    # Medication suggestion considering patient information & 3h's vital measurements
    def __init__(self, config):
        super().__init__(config)

        self.load_ae_physio(config['ae_model_path'])

        self.encoder_physio_addon = nn.Sequential(
            nn.Linear(in_features=self.n_emb + self.n_emb_info,
                      out_features=self.n_emb),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
        )

        self.med_clf = nn.Sequential(
            nn.Linear(in_features=self.n_emb, out_features=self.n_emb // 2),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb // 2, out_features=self.n_emb // 8),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb // 8, out_features=3),
        )

        self.loss_clf = nn.BCEWithLogitsLoss()

    def load_ae_physio(self, model_path):
        files = os.listdir(model_path)
        model_file = None
        model_config = None
        for file in files:
            if "val_loss" in file and 'pred' not in file:
                model_file = os.path.join(model_path, file)
            elif file == "model_config.p":
                model_config = pickle_load(os.path.join(model_path, file))
        self.ae_physio = AE_MED_PHYSIO.load_from_checkpoint(checkpoint_path=model_file, config=model_config)
        self.ae_physio.to(self.device)
        self.ae_physio.train()


    def forward(self, data):
        _, _, x_enc, _, x_info_enc = self.ae_physio(data)
        batchsize = x_enc.size(0)
        # device = x_enc.device

        x_enc_ = self.encoder_physio_addon(torch.concat((x_enc.reshape(batchsize, -1), x_info_enc), dim=1))
        logit = self.med_clf(x_enc_)
        prob = F.sigmoid(logit)

        return logit, prob


    def training_step(self, batch, batch_idx):
        y = batch['med_label'].float()
        logit, prob = self.forward(batch)
        loss = self.loss_clf(logit, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # for metric in self.METRICS:
        #     self.log("train_" + metric, self.METRICS[metric](prob, y.long()), on_step=False, on_epoch=True, prog_bar=True,
        #              logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['med_label'].float()
        logit, prob = self.forward(batch)
        loss = self.loss_clf(logit, y)

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # for metric in self.METRICS:
        #     outputs["val_" + metric] = self.METRICS[metric](prob, y.long())
        #     self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
        #              logger=True)

        return outputs

    def test_step(self, batch, batch_idx):
        y = batch['med_label'].float()
        logit, prob = self.forward(batch)
        loss = self.loss_clf(logit, y)

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # for metric in self.METRICS:
        #     outputs["test_" + metric] = self.METRICS[metric](prob, y.long())
        #     self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
        #              logger=True)

        return outputs



class CONDITION_CLF(BaseCLF):
    # Estimate patient condition based on the vital sign response to vasopressor
    def __init__(self, config):
        super().__init__(config)

        self.n_feat = config['n_feat']
        self.seq_len = config['seq_len']
        self.input_len = self.seq_len // 2

        self.target = config['target']
        if self.target == 'los':
            self.thres_los = config['thres_los']

        self.load_ae_physio(config['ae_model_path'])

        self.encoder_physio_addon = nn.Sequential(
            nn.Linear(in_features=self.n_emb * 2 + self.n_emb_info,
                      out_features=self.n_emb),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
        )

        tcn_params = self.get_tcn_params()
        self.tcn = nn.Sequential(
                TemporalConv1d(in_channels=self.n_feat, out_channels=tcn_params['cnn_out1'],
                               kernel_size=tcn_params['cnn_kernel1'], stride=tcn_params['cnn_stride1'],
                               groups=self.n_feat, causal=False),
                nn.ReLU(),
                TemporalConv1d(in_channels=tcn_params['cnn_out1'], out_channels=tcn_params['cnn_out2'],
                               kernel_size=tcn_params['cnn_kernel2'], stride=tcn_params['cnn_stride2'],
                               groups=1, causal=False),
                nn.ReLU(),
            )

        self.clf = nn.Sequential(
            nn.Linear(in_features=self.n_emb * 2, out_features=self.n_emb),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb, out_features=self.n_emb // 2),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb // 2, out_features=self.n_emb // 8),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb // 8, out_features=1),
        )

    def load_ae_physio(self, model_path):
        files = os.listdir(model_path)
        model_file = None
        model_config = None
        for file in files:
            if "val_loss" in file and 'pred' not in file:
                model_file = os.path.join(model_path, file)
            elif file == "model_config.p":
                model_config = pickle_load(os.path.join(model_path, file))
        self.ae_physio = AE_MED_PHYSIO.load_from_checkpoint(checkpoint_path=model_file, config=model_config)
        self.ae_physio.to(self.device)
        self.ae_physio.eval()

    def get_tcn_params(self):
        cnn_params = {}
        default_params = {
            'cnn_out1': 28,
            'cnn_out2': 14,
            'cnn_kernel1': 4,
            'cnn_kernel2': 8,
            'cnn_stride1': 2,
            'cnn_stride2': 4,
        }
        for param in default_params:
            if param in self.config:
                cnn_params[param] = self.config[param]
            else:
                cnn_params[param] = default_params[param]
        return cnn_params

    def forward(self, data):
        x_next = data['data'][:, self.input_len:, :].float()

        x_hat, x_next_hat, x_enc, x_enc_next, x_info_enc = self.ae_physio(data)
        batchsize = x_enc.size(0)
        # device = x_enc.device

        x_enc_ = self.encoder_physio_addon(torch.concat((x_enc.reshape(batchsize, -1),
                                                         x_enc_next.reshape(batchsize, -1),
                                                         x_info_enc), dim=1))
        x_next_diff = x_next - x_next_hat
        x_diff_enc = self.tcn(x_next_diff.permute(0, 2, 1))
        x_diff_enc = x_diff_enc.reshape(batchsize, -1)

        logit = self.clf(torch.concat((x_enc_, x_diff_enc), dim=1))
        prob = F.sigmoid(logit)

        return logit, prob


    def training_step(self, batch, batch_idx):
        if self.target == 'discharge_status':
            y = batch['info']['discharge_status'].float()
            y = 1 - y
        elif self.target == 'circ_failure':
            y = batch['circ_failure']
            y = (y.sum(dim=1) > 0).float()
        elif self.target == 'los':
            y = batch['los']
            y = y[:, -1]
            y = (y <= self.thres_los).long().float()

        logit, prob = self.forward(batch)

        y = y.reshape(logit.shape)
        loss = self.loss_clf(logit, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](prob, y.long()), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.target == 'discharge_status':
            y = batch['info']['discharge_status'].float()
            y = 1 - y
        elif self.target == 'circ_failure':
            y = batch['circ_failure']
            y = (y.sum(dim=1) > 0).float()
        elif self.target == 'los':
            status = batch['info']['discharge_status']
            idx_remain = status == 1
            for k in batch:
                if k == 'info':
                    for kk in batch[k]:
                        batch[k][kk] = batch[k][kk][idx_remain]
                else:
                    batch[k] = batch[k][idx_remain]
            y = batch['los']
            y = y[:, -1]
            y = (y <= self.thres_los).long().float()


        logit, prob = self.forward(batch)

        y = y.reshape(logit.shape)
        loss = self.loss_clf(logit, y)

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](prob, y.long())
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)

        return outputs

    def test_step(self, batch, batch_idx):
        if self.target == 'discharge_status':
            y = batch['info']['discharge_status'].float()
            y = 1 - y
        elif self.target == 'circ_failure':
            y = batch['circ_failure']
            y = (y.sum(dim=1) > 0).float()
        elif self.target == 'los':
            y = batch['los']
            y = y[:, -1]
            y = (y <= self.thres_los).long().float()

        logit, prob = self.forward(batch)

        y = y.reshape(logit.shape)
        loss = self.loss_clf(logit, y)

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](prob, y.long())
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs



class CONDITION_CLF_TCN(BaseCLF):
    # Estimate patient condition based on the vital sign response to vasopressor
    def __init__(self, config):
        super().__init__(config)

        self.n_feat = config['n_feat']
        self.seq_len = config['seq_len']
        self.input_len = self.seq_len // 2

        self.target = config['target']
        if self.target == 'los':
            self.thres_los = config['thres_los']

        tcn_params = self.get_tcn_params()
        self.tcn = nn.Sequential(
                TemporalConv1d(in_channels=self.n_feat, out_channels=tcn_params['cnn_out1'],
                               kernel_size=tcn_params['cnn_kernel1'], stride=tcn_params['cnn_stride1'],
                               groups=self.n_feat, causal=False),
                nn.ReLU(),
                TemporalConv1d(in_channels=tcn_params['cnn_out1'], out_channels=tcn_params['cnn_out2'],
                               kernel_size=tcn_params['cnn_kernel2'], stride=tcn_params['cnn_stride2'],
                               groups=1, causal=False),
                nn.ReLU(),
            )

        self.clf = nn.Sequential(
            nn.LazyLinear(out_features=self.n_emb),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb, out_features=self.n_emb // 2),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb // 2, out_features=self.n_emb // 8),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb // 8, out_features=1),
        )

    def load_ae_physio(self, model_path):
        files = os.listdir(model_path)
        model_file = None
        model_config = None
        for file in files:
            if "val_loss" in file and 'pred' not in file:
                model_file = os.path.join(model_path, file)
            elif file == "model_config.p":
                model_config = pickle_load(os.path.join(model_path, file))
        self.ae_physio = AE_MED_PHYSIO.load_from_checkpoint(checkpoint_path=model_file, config=model_config)
        self.ae_physio.to(self.device)
        self.ae_physio.eval()

    def get_tcn_params(self):
        cnn_params = {}
        default_params = {
            'cnn_out1': 28,
            'cnn_out2': 14,
            'cnn_kernel1': 4,
            'cnn_kernel2': 8,
            'cnn_stride1': 2,
            'cnn_stride2': 4,
        }
        for param in default_params:
            if param in self.config:
                cnn_params[param] = self.config[param]
            else:
                cnn_params[param] = default_params[param]
        return cnn_params

    def forward(self, data):
        x = data['data'].permute(0, 2, 1).float()

        x_enc = self.tcn(x)
        logit = self.clf(x_enc.reshape(x.shape[0], -1))
        prob = F.sigmoid(logit)

        return logit, prob


    def training_step(self, batch, batch_idx):
        if self.target == 'discharge_status':
            y = batch['info']['discharge_status'].float()
            y = 1 - y
        elif self.target == 'circ_failure':
            y = batch['circ_failure']
            y = (y.sum(dim=1) > 0).float()
        elif self.target == 'los':
            y = batch['los']
            y = y[:, -1]
            y = (y <= self.thres_los).long().float()

        logit, prob = self.forward(batch)

        y = y.reshape(logit.shape)
        loss = self.loss_clf(logit, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](prob, y.long()), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.target == 'discharge_status':
            y = batch['info']['discharge_status'].float()
            y = 1 - y
        elif self.target == 'circ_failure':
            y = batch['circ_failure']
            y = (y.sum(dim=1) > 0).float()
        elif self.target == 'los':
            status = batch['info']['discharge_status']
            idx_remain = status == 1
            for k in batch:
                if k == 'info':
                    for kk in batch[k]:
                        batch[k][kk] = batch[k][kk][idx_remain]
                else:
                    batch[k] = batch[k][idx_remain]
            y = batch['los']
            y = y[:, -1]
            y = (y <= self.thres_los).long().float()


        logit, prob = self.forward(batch)

        y = y.reshape(logit.shape)
        loss = self.loss_clf(logit, y)

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](prob, y.long())
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)

        return outputs

    def test_step(self, batch, batch_idx):
        if self.target == 'discharge_status':
            y = batch['info']['discharge_status'].float()
            y = 1 - y
        elif self.target == 'circ_failure':
            y = batch['circ_failure']
            y = (y.sum(dim=1) > 0).float()
        elif self.target == 'los':
            y = batch['los']
            y = y[:, -1]
            y = (y <= self.thres_los).long().float()

        logit, prob = self.forward(batch)

        y = y.reshape(logit.shape)
        loss = self.loss_clf(logit, y)

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](prob, y.long())
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs


class CONDITION_CLF_TCN_WITH_MED(CONDITION_CLF_TCN):
    # Estimate patient condition based on the vital sign response to vasopressor
    def __init__(self, config):
        super().__init__(config)

        self.n_feat_med = config['n_feat_med']

        tcn_params = self.get_tcn_params()
        self.tcn_med = nn.Sequential(
                TemporalConv1d(in_channels=self.n_feat_med, out_channels=self.n_feat_med*5,
                               kernel_size=tcn_params['cnn_kernel1'], stride=tcn_params['cnn_stride1'],
                               groups=self.n_feat_med, causal=False),
                nn.ReLU(),
                TemporalConv1d(in_channels=self.n_feat_med*5, out_channels=tcn_params['cnn_out2'],
                               kernel_size=tcn_params['cnn_kernel2'], stride=tcn_params['cnn_stride2'],
                               groups=1, causal=False),
                nn.ReLU(),
            )

    def forward(self, data):
        x = data['data'].permute(0, 2, 1).float()
        med = data['med'][:, :, :self.n_feat_med].permute(0, 2, 1).float()
        med[torch.isnan(med)] = 0

        batchsize = x.shape[0]

        x_enc = self.tcn(x)
        med_enc = self.tcn_med(med)
        logit = self.clf(torch.concat((x_enc.reshape(batchsize, -1), med_enc.reshape(batchsize, -1)), dim=1))
        prob = F.sigmoid(logit)

        return logit, prob




