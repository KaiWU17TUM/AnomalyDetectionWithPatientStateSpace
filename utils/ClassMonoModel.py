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
import matplotlib.pyplot as plt

med_dict = {
    0: [1, 0, 0],
    1: [0, 1, 0],
    2: [0, 0, 1],
}


def mae_loss(x_, x, mask_valid=None, ignore_mask=False, ignore_zeros=False, return_arr=False):
    if not ignore_mask:
        if mask_valid is None:
            mask_valid = torch.logical_and((x != -1), (~torch.isnan(x)))
        if ignore_zeros:
            mask_valid = torch.logical_and(mask_valid, (x != 0))
        loss = torch.abs(x[mask_valid] - x_[mask_valid])
    else:
        loss = torch.abs(x - x_)
    if return_arr:
        return loss
    else:
        return torch.mean(loss)


def mse_loss(x_, x, mask_valid=None, ignore_mask=False, ignore_zeros=False, return_arr=False):
    if not ignore_mask:
        if mask_valid is None:
            mask_valid = torch.logical_and((x != -1), (~torch.isnan(x)))
        if ignore_zeros:
            mask_valid = torch.logical_and(mask_valid, (x != 0))
        loss = (x[mask_valid] - x_[mask_valid]) ** 2
    else:
        loss = (x - x_) ** 2

    if return_arr:
        return loss
    else:
        return torch.mean(loss)



class BASE_MODEL(LightningModule):
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
        try:
            self.encoder_type = config['encoder_type']
            self.n_emb = config['n_emb']
        except:
            pass
        try:
            self.n_feat_med = config['n_feat_med']
        except:
            pass
        self.n_emb_info = config['n_emb_info']
        self.dropout = config['dropout']

    def configure_optimizers(self):
        adam = optim.Adam(self.parameters(), lr=self.config['lr'])
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, mode='min', factor=.5, patience=5)
        return [adam], {"scheduler": lr_scheduler, "monitor": "train_loss"}
        # return adam

    def training_epoch_end(self, outputs):
        lr_sch = self.lr_schedulers()
        lr_sch.step(self.trainer.callback_metrics["train_loss"])

    def init_encoder(self):
        if self.encoder_type == 'CNN':
            cnn_params = self.get_cnn_params()
            self.encoder_ae = nn.Sequential(
                nn.Conv1d(in_channels=self.n_feat, out_channels=cnn_params['cnn_out1'],
                          kernel_size=cnn_params['cnn_kernel1'], stride=cnn_params['cnn_stride1'], groups=self.n_feat),
                nn.ReLU(),
                nn.Conv1d(in_channels=cnn_params['cnn_out1'], out_channels=cnn_params['cnn_out2'],
                          kernel_size=cnn_params['cnn_kernel2'], stride=cnn_params['cnn_stride2'], groups=1),
                nn.ReLU(),
            )
            self.decoder_ae = nn.Sequential(
                nn.ConvTranspose1d(in_channels=cnn_params['cnn_out2'], out_channels=cnn_params['cnn_out1'],
                                   kernel_size=cnn_params['cnn_kernel2'], stride=cnn_params['cnn_stride2'], groups=1),
                nn.ReLU(),
                nn.ConvTranspose1d(in_channels=cnn_params['cnn_out1'], out_channels=self.n_feat,
                                   kernel_size=cnn_params['cnn_kernel1'], stride=cnn_params['cnn_stride1'],
                                   groups=self.n_feat),
                nn.Sigmoid(),
            )
            self.decoder_pred = nn.Sequential(
                nn.ConvTranspose1d(in_channels=14, out_channels=56, kernel_size=3, stride=2, groups=1),
                nn.ReLU(),
                nn.ConvTranspose1d(in_channels=56, out_channels=7, kernel_size=6, stride=3, groups=7),
                nn.Sigmoid(),
            )

        elif self.encoder_type == 'TCN':
            tcn_params = self.get_tcn_params()
            self.encoder_ae = nn.Sequential(
                TemporalConv1d(in_channels=self.n_feat, out_channels=tcn_params['cnn_out1'],
                               kernel_size=tcn_params['cnn_kernel1'], stride=tcn_params['cnn_stride1'],
                               groups=self.n_feat, causal=False),
                nn.ReLU(),
                TemporalConv1d(in_channels=tcn_params['cnn_out1'], out_channels=tcn_params['cnn_out2'],
                               kernel_size=tcn_params['cnn_kernel2'], stride=tcn_params['cnn_stride2'],
                               groups=1, causal=False),
                nn.ReLU(),
            )
            self.decoder_ae = nn.Sequential(
                TemporalConvTranspose1d(in_channels=tcn_params['cnn_out2'], out_channels=tcn_params['cnn_out1'],
                                        kernel_size=tcn_params['cnn_kernel2'], stride=tcn_params['cnn_stride2'],
                                        groups=1, causal=False),
                nn.ReLU(),
                TemporalConvTranspose1d(in_channels=tcn_params['cnn_out1'], out_channels=self.n_feat,
                                        kernel_size=tcn_params['cnn_kernel1'], stride=tcn_params['cnn_stride1'],
                                        groups=self.n_feat, causal=False),
                nn.Sigmoid(),
            )
            self.decoder_pred = nn.Sequential(
                TemporalConvTranspose1d(in_channels=tcn_params['cnn_out2'], out_channels=tcn_params['cnn_out1'],
                                        kernel_size=tcn_params['cnn_kernel2'], stride=tcn_params['cnn_stride2'],
                                        groups=1, causal=False),
                nn.ReLU(),
                TemporalConvTranspose1d(in_channels=tcn_params['cnn_out1'], out_channels=self.n_feat,
                                        kernel_size=tcn_params['cnn_kernel1'], stride=tcn_params['cnn_stride1'],
                                        groups=self.n_feat, causal=False),
                nn.Sigmoid(),
            )
        elif self.encoder_type == 'LSTM':
            pass
        elif self.encoder_type == 'ATT':
            pass
        else:
            pass

        self.encoder_info = nn.Sequential(
            nn.LazyLinear(out_features=self.n_emb_info),
            nn.ReLU()
        )
        self.encoder_physio_addon = nn.Sequential(
            nn.Linear(in_features=self.n_emb + self.n_emb_info, out_features=self.n_emb)
        )

    def get_cnn_params(self):
        cnn_params = {}
        default_params = {
            'cnn_out1': 56,
            'cnn_out2': 14,
            'cnn_kernel1': 6,
            'cnn_kernel2': 3,
            'cnn_stride1': 3,
            'cnn_stride2': 2,
        }
        for param in default_params:
            if param in self.config:
                cnn_params[param] = self.config[param]
            else:
                cnn_params[param] = default_params[param]
        return cnn_params

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


class AE_PHYSIO(BASE_MODEL):
    def __init__(self, config):
        super().__init__(config)

        self.init_encoder()

        self.state_transition = nn.Sequential(
            nn.Linear(in_features=self.n_emb + self.n_emb_info, out_features=self.n_emb),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb, out_features=self.n_emb),
        )

        self.METRICS = {
            'mse': mse_loss,
        }
        self.loss_weight_constrait = nn.MSELoss()
        # self.config = config
        # # input params
        # self.batchsize = config['batchsize']
        # self.seq_len = config['seq_len']
        # self.n_feat = config['n_feat']
        # self.input_len = int(self.seq_len // 2)
        # self.pred_len = int(self.seq_len // 2)
        # # model params
        # self.encoder_type = config['encoder_type']
        # self.n_emb = config['n_emb']
        # self.n_emb_info = config['n_emb_info']
        # self.dropout = config['dropout']
        #
        # self.init_encoder()
        #
        # self.state_transition = nn.Sequential(
        #     nn.Linear(in_features=self.n_emb + self.n_emb_info, out_features=self.n_emb),
        #     nn.ReLU(),
        #     nn.Linear(in_features=self.n_emb, out_features=self.n_emb),
        # )
        #
        # self.METRICS = {
        #     'mse': mse_loss,
        # }
        # self.loss_weight_constrait = nn.MSELoss()

    def forward(self, data):
        age = data['info']['age']
        height = data['info']['height']
        sex = data['info']['sex']
        apache = data['info']['apache']
        x_info = torch.cat((age, height, sex, apache), dim=1)
        x = data['data'][:, :self.input_len, :].permute(0, 2, 1).float()

        # Encode time-seires vital signs
        x_enc = self.encoder_ae(x)
        # print(f"111 X_ENC: {x_enc.shape}")
        x_hat = self.decoder_ae(x_enc)
        # print(f"222 X_DEC: {x_hat.shape}")

        # Combine patient info with vital signs
        x_info_enc = self.encoder_info(x_info)
        # print(f"333 INFO ENC: {x_info_enc.shape}")
        x_enc_next = self.state_transition(torch.concat((x_enc.reshape(x_enc.shape[0], -1), x_info_enc), dim=1))
        # print(f"444 X_ENC_NEXT: {x_enc_next.shape}")
        x_next_hat = self.decoder_pred(x_enc_next.reshape(x_enc.shape))
        # print(f"555 X_NEXT_HAT: {x_next_hat.shape}")

        if self.encoder_type == 'TCN':
            pad_len = int((x_hat.shape[2] - self.pred_len) // 2)
            x_hat = x_hat[:, :, pad_len:x_hat.shape[2] - pad_len]
            x_next_hat = x_next_hat[:, :, pad_len:x_next_hat.shape[2] - pad_len]
            # print(f"666 X_HAT: {x_hat.shape}, X_NEXT_HAT: {x_next_hat.shape}")
        return x_hat.permute(0, 2, 1), x_next_hat.permute(0, 2, 1), x_enc, x_enc_next, x_info_enc

    def training_step(self, batch, batch_idx):
        x = batch['data']
        x_mask = batch['data_mask']
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_hat, x_next_hat, x_enc, x_enc_next, x_info_enc = self.forward(batch)
        # print(torch.isnan(x.reshape(-1)).sum().item(),
        #       torch.isnan(x_hat.reshape(-1)).sum().item(),
        #       torch.isnan(x_next_hat.reshape(-1)).sum().item())

        loss_ae = mae_loss(x_hat, x_curr, x_mask[:, :90, :])
        loss_pred = mae_loss(x_next_hat, x_next, x_mask[:, 90:, :])
        loss_similarity_dec = self.loss_weight_constrait(self.decoder_ae[0].weight,
                                                         self.decoder_pred[0].weight) \
                              + self.loss_weight_constrait(self.decoder_ae[2].weight,
                                                           self.decoder_pred[2].weight)

        # loss = self.alpha * loss_pred + self.beta * loss_ae
        loss = self.alpha * loss_pred + self.beta * loss_ae + loss_similarity_dec

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_pred, loss_similarity_dec],
                                    ['loss_reconst', 'loss_pred', 'loss_constrain']):
            self.log("train_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        x_mask = batch['data_mask']
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_hat, x_next_hat, x_enc, x_enc_next, x_info_enc = self.forward(batch)

        loss_ae = mae_loss(x_hat, x_curr, x_mask[:, :90, :])
        loss_pred = mae_loss(x_next_hat, x_next, x_mask[:, 90:, :])
        loss_similarity_dec = self.loss_weight_constrait(self.decoder_ae[0].weight,
                                                         self.decoder_pred[0].weight) \
                              + self.loss_weight_constrait(self.decoder_ae[2].weight,
                                                           self.decoder_pred[2].weight)

        # loss = self.alpha * loss_pred + self.beta * loss_ae
        loss = self.alpha * loss_pred + self.beta * loss_ae + loss_similarity_dec

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_pred, loss_similarity_dec],
                                    ['loss_reconst', 'loss_pred', 'loss_constrain']):
            self.log("val_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']
        x_mask = batch['data_mask']
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_hat, x_next_hat, x_enc, x_enc_next, x_info_enc = self.forward(batch)

        loss_ae = mae_loss(x_hat, x_curr, x_mask[:, :90, :])
        loss_pred = mae_loss(x_next_hat, x_next, x_mask[:, 90:, :])
        loss_similarity_dec = self.loss_weight_constrait(self.decoder_ae[0].weight,
                                                         self.decoder_pred[0].weight) \
                              + self.loss_weight_constrait(self.decoder_ae[2].weight,
                                                           self.decoder_pred[2].weight)

        # loss = self.alpha * loss_pred + self.beta * loss_ae
        loss = self.alpha * loss_pred + self.beta * loss_ae + loss_similarity_dec

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_pred, loss_similarity_dec],
                                    ['loss_reconst', 'loss_pred', 'loss_constrain']):
            self.log("test_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return outputs


class AE_MED_PHYSIO(BASE_MODEL):
    def __init__(self, config):
        super().__init__(config)

        self.init_encoder()
        if config['encoder_type'] == 'CNN':
            cnn_params = self.get_cnn_params()
            self.encoder_med = nn.Sequential(
                nn.Conv1d(in_channels=self.n_feat_med, out_channels=self.n_feat_med * 5,
                          kernel_size=cnn_params['cnn_kernel1'], stride=cnn_params['cnn_stride1'], groups=self.n_feat_med),
                nn.ReLU(),
                nn.Conv1d(in_channels=self.n_feat_med * 5, out_channels=cnn_params['cnn_out2'],
                          kernel_size=cnn_params['cnn_kernel2'], stride=cnn_params['cnn_stride2'], groups=1),
                nn.ReLU(),
            )
        elif config['encoder_type'] == 'TCN':
            tcn_params = self.get_tcn_params()
            self.encoder_med = nn.Sequential(
                TemporalConv1d(in_channels=self.n_feat_med, out_channels=self.n_feat_med * 5,
                               kernel_size=tcn_params['cnn_kernel1'], stride=tcn_params['cnn_stride1'],
                               groups=self.n_feat_med, causal=False),
                nn.ReLU(),
                TemporalConv1d(in_channels=self.n_feat_med * 5, out_channels=tcn_params['cnn_out2'],
                               kernel_size=tcn_params['cnn_kernel2'], stride=tcn_params['cnn_stride2'],
                               groups=1, causal=False),
                nn.ReLU(),
            )

        self.state_transition = nn.Sequential(
            nn.Linear(in_features=self.n_emb + self.n_emb_info, out_features=self.n_emb),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb, out_features=self.n_emb),
        )
        self.control_input = nn.Sequential(
            nn.Linear(in_features=self.n_emb + self.n_emb_info, out_features=self.n_emb),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb, out_features=self.n_emb),
        )


        self.METRICS = {
            'mse': mse_loss,
        }
        self.loss_weight_constrait = nn.MSELoss()


    def forward(self, data):
        age = data['info']['age']
        height = data['info']['height']
        sex = data['info']['sex']
        apache = data['info']['apache']
        x_info = torch.cat((age, height, sex, apache), dim=1)
        x = data['data'][:, :self.input_len, :].permute(0, 2, 1).float()
        med = data['med'][:, self.input_len:, :self.n_feat_med].permute(0, 2, 1).float()
        med[torch.isnan(med)] = 0

        # Encode time-seires vital signs
        x_enc = self.encoder_ae(x)
        # print(f"111 X_ENC: {x_enc.shape}")
        x_hat = self.decoder_ae(x_enc)
        # print(f"222 X_DEC: {x_hat.shape}")

        # Combine patient info with vital signs
        x_info_enc = self.encoder_info(x_info)

        med_enc = self.encoder_med(med)
        # print(f"333 MED ENC: {med_enc.shape}")

        # mimic Kalman filter
        x_enc_next = self.state_transition(torch.concat((x_enc.reshape(x_enc.shape[0], -1), x_info_enc), dim=1)) \
                     + self.control_input(torch.concat((med_enc.reshape(x_enc.shape[0], -1), x_info_enc), dim=1))


        # print(f"333 INFO ENC: {x_info_enc.shape}")
        # print(f"444 X_ENC_NEXT: {x_enc_next.shape}")
        x_next_hat = self.decoder_pred(x_enc_next.reshape(x_enc.shape))
        # print(f"555 X_NEXT_HAT: {x_next_hat.shape}")

        if self.encoder_type == 'TCN':
            pad_len = int((x_hat.shape[2] - self.pred_len) // 2)
            x_hat = x_hat[:, :, pad_len:x_hat.shape[2] - pad_len]
            x_next_hat = x_next_hat[:, :, pad_len:x_next_hat.shape[2] - pad_len]
            # print(f"666 X_HAT: {x_hat.shape}, X_NEXT_HAT: {x_next_hat.shape}")
        return x_hat.permute(0, 2, 1), x_next_hat.permute(0, 2, 1), x_enc, x_enc_next, x_info_enc

    def training_step(self, batch, batch_idx):
        x = batch['data']
        x_mask = batch['data_mask']
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_hat, x_next_hat, x_enc, x_enc_next, x_info_enc = self.forward(batch)
        # print(torch.isnan(x.reshape(-1)).sum().item(),
        #       torch.isnan(x_hat.reshape(-1)).sum().item(),
        #       torch.isnan(x_next_hat.reshape(-1)).sum().item())

        loss_ae = mae_loss(x_hat, x_curr, x_mask[:, :90, :])
        loss_pred = mae_loss(x_next_hat, x_next, x_mask[:, 90:, :])
        loss_similarity_dec = self.loss_weight_constrait(self.decoder_ae[0].weight,
                                                         self.decoder_pred[0].weight) \
                              + self.loss_weight_constrait(self.decoder_ae[2].weight,
                                                           self.decoder_pred[2].weight)

        # loss = self.alpha * loss_pred + self.beta * loss_ae
        loss = loss_pred + loss_ae + 0.2 * loss_similarity_dec

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_pred, loss_similarity_dec],
                                    ['loss_reconst', 'loss_pred', 'loss_constrain']):
            self.log("train_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        x_mask = batch['data_mask']
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_hat, x_next_hat, x_enc, x_enc_next, x_info_enc = self.forward(batch)

        loss_ae = mae_loss(x_hat, x_curr, x_mask[:, :90, :])
        loss_pred = mae_loss(x_next_hat, x_next, x_mask[:, 90:, :])
        loss_similarity_dec = self.loss_weight_constrait(self.decoder_ae[0].weight,
                                                         self.decoder_pred[0].weight) \
                              + self.loss_weight_constrait(self.decoder_ae[2].weight,
                                                           self.decoder_pred[2].weight)

        # loss = self.alpha * loss_pred + self.beta * loss_ae
        loss = loss_pred + loss_ae + 0.2 * loss_similarity_dec

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_pred, loss_similarity_dec],
                                    ['loss_reconst', 'loss_pred', 'loss_constrain']):
            self.log("val_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']
        x_mask = batch['data_mask']
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_hat, x_next_hat, x_enc, x_enc_next, x_info_enc = self.forward(batch)

        loss_ae = mae_loss(x_hat, x_curr, x_mask[:, :90, :])
        loss_pred = mae_loss(x_next_hat, x_next, x_mask[:, 90:, :])
        loss_similarity_dec = self.loss_weight_constrait(self.decoder_ae[0].weight,
                                                         self.decoder_pred[0].weight) \
                              + self.loss_weight_constrait(self.decoder_ae[2].weight,
                                                           self.decoder_pred[2].weight)

        # loss = self.alpha * loss_pred + self.beta * loss_ae
        loss = loss_pred + loss_ae + 0.2 * loss_similarity_dec

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_pred, loss_similarity_dec],
                                    ['loss_reconst', 'loss_pred', 'loss_constrain']):
            self.log("test_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return outputs


class MED_PHYSIO_MONO(BASE_MODEL):
    def __init__(self, config):
        super().__init__(config)
        self.n_feat_med = config['n_feat_med']
        self.n_emb_mono = config['n_emb_mono']
        self.n_groupsort = config['n_groupsort']
        self.monotonic_constraints = config['monotonic_constraints']

        self.load_ae_physio(config['model_path_ae'])
        # self.state_transition = nn.Sequential(
        #     nn.Linear(in_features=self.n_emb + self.n_emb_info, out_features=self.n_emb),
        #     nn.ReLU(),
        #     nn.Linear(in_features=self.n_emb, out_features=self.n_emb),
        # )
        self.monotonic_layer = lmn.MonotonicWrapper(
            nn.Sequential(
                lmn.LipschitzLinear(self.n_feat_med + self.n_emb, self.n_emb_mono, kind="one-inf"),
                lmn.GroupSort(self.n_groupsort),
                lmn.LipschitzLinear(self.n_emb_mono, self.n_feat, kind="inf"),
            ),
            monotonic_constraints=self.monotonic_constraints.astype(np.double)
        )

        self.METRICS = {
            'mse': mse_loss,
        }

    def load_ae_physio(self, model_path):
        files = os.listdir(model_path)
        model_file = None
        model_config = None
        for file in files:
            if "val_loss_pred" in file:
                model_file = os.path.join(model_path, file)
            elif file == "model_config.p":
                model_config = pickle_load(os.path.join(model_path, file))
        self.ae_physio = AE_PHYSIO.load_from_checkpoint(checkpoint_path=model_file, config=model_config)
        self.ae_physio.to(self.device)
        # self.ae_physio.eval()

    def forward(self, data):
        age = data['info']['age']
        height = data['info']['height']
        sex = data['info']['sex']
        apache = data['info']['apache']
        x_info = torch.cat((age, height, sex, apache), dim=1)
        x = data['data'][:, :self.input_len, :].permute(0, 2, 1).float()

        x_hat, x_next_hat, x_enc, x_enc_next, x_info_enc = self.ae_physio(data)
        # print(f"000 AE OUTPUT: X_ENC - {x_enc.shape}, X_HAT - {x_hat.shape}")

        x_med = data['med'][:, :self.input_len, :].permute(0, 2, 1).float()
        x_med[torch.isnan(x_med)] = 0

        # Predict future vital sign with monotonic constraints on medication
        # print(f"CHECK: {x_enc_next.reshape(x_enc_next.shape[0], -1).shape}   {x_med[:,:,0].shape}   {torch.concat((x_enc_next.reshape(x_enc_next.shape[0], -1), x_med[:,:,0]), dim=1).shape}")
        x_next_hat = self.monotonic_layer(
            torch.concat((x_enc_next.reshape(x_enc_next.shape[0], -1), x_med[:, :, 0]), dim=1)
        )
        # print(f"111 MONO OUTPUT: X_NEXT_HAT - {x_next_hat.shape}")
        x_next_hat = x_next_hat[:, :, None]
        # print(f"333 INFO X_NEXT_HAT: {x_next_hat.shape}")
        for i in range(1, self.pred_len):
            if i == 0:
                x_enc_ = x_enc_next.copy()
            else:
                x_curr = torch.concat((x[:, :, i:], x_next_hat), dim=2)
                x_enc = self.ae_physio.encoder_ae(x_curr)
                x_enc_ = self.ae_physio.state_transition(
                    torch.concat((x_enc.reshape(x_enc.shape[0], -1), x_info_enc), dim=1))

            x_next = self.monotonic_layer(
                torch.concat((x_enc_.reshape(x_enc_.shape[0], -1), x_med[:, :, i]), dim=1)
            )
            x_next_hat = torch.concat((x_next_hat, x_next[:, :, None]), dim=2)
        # print(f"333 INFO X_NEXT_HAT: {x_next_hat.shape}")

        return x_next_hat.permute(0, 2, 1)

    def training_step(self, batch, batch_idx):
        x = batch['data']
        x_mask = batch['data_mask']
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_next_hat = self.forward(batch)
        # print(torch.isnan(x.reshape(-1)).sum().item(),
        #       torch.isnan(x_hat.reshape(-1)).sum().item(),
        #       torch.isnan(x_next_hat.reshape(-1)).sum().item())

        # loss_ae = mae_loss(x_hat, x_curr, x_mask[:, :90, :])
        loss_pred = mae_loss(x_next_hat, x_next, x_mask[:, 90:, :])

        # loss = self.alpha * loss_pred + self.beta * loss_ae
        loss = loss_pred

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_pred], ['loss_pred']):
            self.log("train_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        x_mask = batch['data_mask']
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_next_hat = self.forward(batch)

        # loss_ae = mae_loss(x_hat, x_curr, x_mask[:, :90, :])
        loss_pred = mae_loss(x_next_hat, x_next, x_mask[:, 90:, :])

        # loss = self.alpha * loss_pred + self.beta * loss_ae
        loss = loss_pred

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_pred], ['loss_pred']):
            self.log("val_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']
        x_mask = batch['data_mask']
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_next_hat = self.forward(batch)

        # loss_ae = mae_loss(x_hat, x_curr, x_mask[:, :90, :])
        loss_pred = mae_loss(x_next_hat, x_next, x_mask[:, 90:, :])

        # loss = self.alpha * loss_pred + self.beta * loss_ae
        loss = loss_pred

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_pred], ['loss_pred']):
            self.log("test_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return outputs






class MED_ITERATIVE_MONO(BASE_MODEL):
    def __init__(self, config):
        super().__init__(config)
        self.n_feat_med = config['n_feat_med']
        self.n_step = config['n_step']
        self.n_step_med = config['n_step_med']
        self.n_emb_info = config['n_emb_info']
        # self.load_ae_physio(config['model_path_ae'])
        self.regression_type = config['regression_type']
        self.n_emb_mono = config['n_emb_mono']
        self.n_groupsort = config['n_groupsort']
        self.monotonic_constraints_physio = config['monotonic_constraints_physio']
        # self.monotonic_constraints_med = config['monotonic_constraints_med']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.last_epoch = 0

        self.regression_layer = nn.Sequential(
            nn.Linear(in_features=self.n_step * self.n_feat, out_features=self.n_step * self.n_feat),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.n_step * self.n_feat, out_features=self.n_feat),
        )
        self.emb_info = nn.Sequential(
            nn.LazyLinear(out_features=self.n_emb_info),
            nn.LeakyReLU(),
        )

        if self.regression_type == 'mono':
            # input: current vital + pred vital + current med + accumulated med + embedded info
            self.monotonic_physio = lmn.MonotonicWrapper(
                nn.Sequential(
                    lmn.LipschitzLinear(2 * self.n_feat + 2 * self.n_feat_med + self.n_emb_info,
                                        2 * self.n_feat + 2 * self.n_feat_med + self.n_emb_info, kind="one-inf"),
                    nn.LeakyReLU(),
                    # lmn.GroupSort(self.n_groupsort),
                    lmn.LipschitzLinear(2 * self.n_feat + 2 * self.n_feat_med + self.n_emb_info, self.n_feat, kind="inf"),
                ),
                monotonic_constraints=self.monotonic_constraints_physio.astype(np.double)
            )
        elif self.regression_type == 'fc':
            self.monotonic_physio = nn.Sequential(
                    nn.Linear(2 * self.n_feat + 2 * self.n_feat_med + self.n_emb_info, 2 * self.n_feat + 2 * self.n_feat_med + self.n_emb_info),
                    nn.LeakyReLU(),
                    nn.Linear(2 * self.n_feat + 2 * self.n_feat_med + self.n_emb_info, self.n_feat),
                )
        else:
            raise ValueError(f"Unsupported regression type: {self.regression_type}. ")
        # # input: diff(next vital, pred vital) + next vital + current med + accumulated med + embedded info
        # self.monotonic_med = lmn.MonotonicWrapper(
        #     nn.Sequential(
        #         lmn.LipschitzLinear(2 * self.n_feat + 2 * self.n_feat_med + self.n_emb_info, self.n_feat_med, kind="one-inf"),
        #         nn.LeakyReLU(),
        #         lmn.LipschitzLinear(self.n_feat_med, 1, kind="one-inf"),
        #         # lmn.GroupSort(self.n_groupsort),
        #         # lmn.LipschitzLinear(self.n_emb_mono, self.n_feat, kind="inf"),
        #     ),
        #     monotonic_constraints=self.monotonic_constraints_med.astype(np.double)
        # )

        self.METRICS = {
            'mse': mse_loss,
        }
        loss_mse_nozeros = functools.partial(mse_loss, ignore_mask=True, return_arr=True)
        self.loss_soft_dtw = SoftDTWLossPyTorch(gamma=0.1, normalize=True,
                                                dist_func=loss_mse_nozeros)

    def forward(self, data):
        # patient information
        age = data['info']['age']
        height = data['info']['height']
        sex = data['info']['sex']
        apache = data['info']['apache']
        x_info = torch.cat((age, height, sex, apache), dim=1)
        # time-series vital sign
        x = data['data'].float()
        if self.n_feat == 1:
            x = x[:, :, 4][:, :, None] # MAP
        # time-series infusion data
        med_label = data['med_label']
        x_med = data['med'][:,:,:3].float()
        x_med_acc = data['med_acc'][:,:,:3].float()
        x_med[torch.isnan(x_med)] = 0
        dosage_trend = data['dosage_trend'].float()
        dosage_trend_bool = data['dosage_trend_bool']

        # Predict future vital sign with monotonic constraints on medication
        # print(f"CHECK: {x_enc_next.reshape(x_enc_next.shape[0], -1).shape}   {x_med[:,:,0].shape}   {torch.concat((x_enc_next.reshape(x_enc_next.shape[0], -1), x_med[:,:,0]), dim=1).shape}")

        x_info_emb = self.emb_info(x_info)
        x_next_ar = torch.empty((x.shape[0], 0), dtype=torch.float32).to(self.device)
        x_delta_med = torch.empty((x.shape[0], 0), dtype=torch.float32).to(self.device)
        x_next_hat = torch.empty((x.shape[0], 0), dtype=torch.float32).to(self.device)
        for i in range(self.n_step_med, self.seq_len-self.n_step + 1):
            x_curr = x[:, i:i+self.n_step, :]
            try:
                x_next_ = self.regression_layer(x_curr.reshape(x.shape[0], -1))
            except:
                print(111)
            med_curr = x_med[:, i, :]
            med_acc = x_med_acc[:, i, :]
            x_next_delta = self.monotonic_physio(torch.cat((x_curr[:,-1,:], x_next_, med_curr, med_acc, x_info_emb), dim=1))
            x_next = x_next_ + x_next_delta
            x_next_ar = torch.concat((x_next_ar, x_next_), dim=1)
            x_delta_med = torch.concat((x_delta_med, x_next_delta), dim=1)
            x_next_hat = torch.concat((x_next_hat, x_next), dim=1)

        x_diff = x_next_hat.diff(dim=1)
        x_diff_acc = x_diff.cumsum(dim=1)

# plt.plot(x_curr.detach().cpu().reshape(16, -1).T);
# plt.show()
# plt.plot(x.detach().cpu().reshape(16, -1).T);
# plt.show()

        # print(f"X_NEXT_HAT: {x_next_hat.shape}")

        return x_next_hat, x_next_ar, x_delta_med, x_diff, x_diff_acc


    def training_step(self, batch, batch_idx):
        # valid segment - [:, self.n_step_med:(self.seq_len - self.n_step//2), :]
        x_regression = batch['data_regression'][:, self.n_step_med:(self.seq_len - self.n_step + 1), :]
        if self.n_feat == 1:
            x_regression = x_regression[:, :, 4]  # MAP

        x_med = batch['med'][:, :, :3].float()[:, self.n_step_med:(self.seq_len - self.n_step + 1), :]
        x_med[torch.isnan(x_med)] = 0
        x_med_acc = batch['med_acc'][:, :, :3].float()[:, self.n_step_med:(self.seq_len - self.n_step + 1), :]
        x_med_acc[torch.isnan(x_med_acc)] = 0

        # print(x_mask.sum())
        x_next_hat, x_next_ar, x_delta_med, x_diff, x_diff_acc = self.forward(batch)

        x_pred1 = x_next_hat[:, :self.seq_len//3-self.n_step_med]
        x_pred2 = x_next_hat[:, self.seq_len//3-self.n_step_med : 2*self.seq_len//3-self.n_step_med]
        x_pred3 = x_next_hat[:, 2*self.seq_len//3-self.n_step_med:]
        x_reg1 = x_regression[:, :self.seq_len//3-self.n_step_med]
        x_reg2 = x_regression[:, self.seq_len//3-self.n_step_med : 2*self.seq_len//3-self.n_step_med]
        x_reg3 = x_regression[:, 2*self.seq_len//3-self.n_step_med:]
        loss_before = mse_loss(x_pred1, x_reg1)
        loss_med_effect = mse_loss(x_pred2, x_reg2)
        loss_after = mse_loss(x_pred3, x_reg3)

        loss_med_constraint = self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med[:, :, 0][:, 75:-1, None]) + \
                              self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med[:, :, 1][:, 75:-1, None]) + \
                              self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med[:, :, 2][:, 75:-1, None])
        loss_med_constraint = loss_med_constraint.mean()

        # loss_med_constraint_acc = self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med_acc[:, :, 0][:, 75:-1, None]) + \
        #                           self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med_acc[:, :, 1][:, 75:-1, None]) + \
        #                           self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med_acc[:, :, 2][:, 75:-1, None])
        # loss_med_constraint_acc = loss_med_constraint_acc.mean()

        loss_ar = loss_before + self.alpha * loss_med_effect + loss_after
        loss = loss_before + self.alpha * loss_med_effect + loss_after + \
            self.beta * loss_med_constraint

        if self.current_epoch != self.last_epoch:
            med_label = batch['med_label'][1]
            fig, ax = plt.subplots(2,1,figsize=(10,6))
            fig.suptitle(f'Epoch: {self.current_epoch} - cumsum')
            ax[0].plot(batch['data'][1, self.n_step_med:(self.seq_len - self.n_step + 1), 4].cpu(), label='raw')
            # ax[0].plot(x_regression[1, :].detach().cpu(), label='regression')
            ax[0].plot(x_next_hat[1, :].detach().cpu(), label='pred')
            ax[0].plot(x_next_ar[1, :].detach().cpu(), label='pred_ar')
            ax[0].plot(x_delta_med[1, :].detach().cpu(), label='med_delta')
            ax[0].plot(x_diff[1, :].detach().cpu(), label='x_delta')
            ax[0].plot(x_diff_acc[1, :].detach().cpu(), label='x_delta_cumsum')
            ax[0].legend()
            ax[1].plot(batch['med'][1, self.n_step_med:(self.seq_len - self.n_step + 1), :3].cpu(), label='med')
            fig.suptitle(f"DTW: {loss_med_constraint:.4f} - AR: {loss_ar:.4f}")
            plt.show()
            self.last_epoch = self.current_epoch

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_med_constraint, loss_ar], ['cstr', 'loss_ar']):
            self.log("train_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # valid segment - [:, self.n_step_med:(self.seq_len - self.n_step//2), :]
        x_regression = batch['data_regression'][:, self.n_step_med:(self.seq_len - self.n_step + 1), :]
        if self.n_feat == 1:
            x_regression = x_regression[:, :, 4]  # MAP

        x_med = batch['med'][:, :, :3].float()[:, self.n_step_med:(self.seq_len - self.n_step + 1), :]
        x_med[torch.isnan(x_med)] = 0
        x_med_acc = batch['med_acc'][:, :, :3].float()[:, self.n_step_med:(self.seq_len - self.n_step + 1), :]
        x_med_acc[torch.isnan(x_med_acc)] = 0

        # print(x_mask.sum())
        x_next_hat, x_next_ar, x_delta_med, x_diff, x_diff_acc = self.forward(batch)

        x_pred1 = x_next_hat[:, :self.seq_len // 3 - self.n_step_med]
        x_pred2 = x_next_hat[:, self.seq_len // 3 - self.n_step_med: 2 * self.seq_len // 3 - self.n_step_med]
        x_pred3 = x_next_hat[:, 2 * self.seq_len // 3 - self.n_step_med:]
        x_reg1 = x_regression[:, :self.seq_len // 3 - self.n_step_med]
        x_reg2 = x_regression[:, self.seq_len // 3 - self.n_step_med: 2 * self.seq_len // 3 - self.n_step_med]
        x_reg3 = x_regression[:, 2 * self.seq_len // 3 - self.n_step_med:]
        loss_before = mse_loss(x_pred1, x_reg1)
        loss_med_effect = mse_loss(x_pred2, x_reg2)
        loss_after = mse_loss(x_pred3, x_reg3)

        loss_med_constraint = self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med[:, :, 0][:, 75:-1, None]) + \
                              self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med[:, :, 1][:, 75:-1, None]) + \
                              self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med[:, :, 2][:, 75:-1, None])
        loss_med_constraint = loss_med_constraint.mean()

        # loss_med_constraint_acc = self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med_acc[:, :, 0][:, 75:-1, None]) + \
        #                           self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med_acc[:, :, 1][:, 75:-1, None]) + \
        #                           self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med_acc[:, :, 2][:, 75:-1, None])
        # loss_med_constraint_acc = loss_med_constraint_acc.mean()

        loss_ar = loss_before + self.alpha * loss_med_effect + loss_after
        loss = loss_before + self.alpha * loss_med_effect + loss_after + \
               self.beta * loss_med_constraint

        outputs = {'val_loss': loss, 'val_loss_ar': loss_ar}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_med_constraint, loss_ar], ['cstr', 'loss_ar']):
            self.log("val_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        # valid segment - [:, self.n_step_med:(self.seq_len - self.n_step//2), :]
        x_regression = batch['data_regression'][:, self.n_step_med:(self.seq_len - self.n_step + 1), :]
        if self.n_feat == 1:
            x_regression = x_regression[:, :, 4]  # MAP

        x_med = batch['med'][:, :, :3].float()[:, self.n_step_med:(self.seq_len - self.n_step + 1), :]
        x_med[torch.isnan(x_med)] = 0
        x_med_acc = batch['med_acc'][:, :, :3].float()[:, self.n_step_med:(self.seq_len - self.n_step + 1), :]
        x_med_acc[torch.isnan(x_med_acc)] = 0

        # print(x_mask.sum())
        x_next_hat, x_next_ar, x_delta_med, x_diff, x_diff_acc = self.forward(batch)

        x_pred1 = x_next_hat[:, :self.seq_len // 3 - self.n_step_med]
        x_pred2 = x_next_hat[:, self.seq_len // 3 - self.n_step_med: 2 * self.seq_len // 3 - self.n_step_med]
        x_pred3 = x_next_hat[:, 2 * self.seq_len // 3 - self.n_step_med:]
        x_reg1 = x_regression[:, :self.seq_len // 3 - self.n_step_med]
        x_reg2 = x_regression[:, self.seq_len // 3 - self.n_step_med: 2 * self.seq_len // 3 - self.n_step_med]
        x_reg3 = x_regression[:, 2 * self.seq_len // 3 - self.n_step_med:]
        loss_before = mse_loss(x_pred1, x_reg1)
        loss_med_effect = mse_loss(x_pred2, x_reg2)
        loss_after = mse_loss(x_pred3, x_reg3)

        loss_med_constraint = self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med[:, :, 0][:, 75:-1, None]) + \
                              self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med[:, :, 1][:, 75:-1, None]) + \
                              self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med[:, :, 2][:, 75:-1, None])
        loss_med_constraint = loss_med_constraint.mean()

        # loss_med_constraint_acc = self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med_acc[:, :, 0][:, 75:-1, None]) + \
        #                           self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med_acc[:, :, 1][:, 75:-1, None]) + \
        #                           self.loss_soft_dtw(x_diff_acc[:, 75:, None], x_med_acc[:, :, 2][:, 75:-1, None])
        # loss_med_constraint_acc = loss_med_constraint_acc.mean()


        loss_ar = loss_before + self.alpha * loss_med_effect + loss_after
        loss = loss_before + self.alpha * loss_med_effect + loss_after + \
               self.beta * loss_med_constraint

        outputs = {'test_loss': loss, 'test_loss_ar': loss_ar}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_med_constraint, loss_ar], ['cstr', 'loss_ar']):
            self.log("test_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return outputs
