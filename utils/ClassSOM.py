import os
import numpy as np
from copy import deepcopy

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torch.autograd as autograd
import monotonicnetworks as lmn
from monotonicnetworks import GroupSort

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.data_io import pickle_load


med_dict = {
    0: [1,0,0],
    1: [0,1,0],
    2: [0,0,1],
}

def mae_loss(x_, x, mask_valid=None):
    if mask_valid is None:
        mask_valid = x!=-1
    loss = torch.mean(torch.abs(x[mask_valid] - x_[mask_valid]))
    return loss

def mse_loss(x_, x, mask_valid=None):
    if mask_valid is None:
        mask_valid = x != -1
    loss = torch.mean((x[mask_valid] - x_[mask_valid])**2)
    return loss

def index_map(n):
    index_map = {i: (i // n, i % n) for i in range(n**2)}
    index_map_invert = {j: i for i, j in index_map.items()}

    return index_map, index_map_invert


class SOM_EMB_FUNC(autograd.Function):
    # map encoded vector x in to a centroid in SOM embedding space
    @staticmethod
    def forward(ctx, x, centroids):
        ###TODO: CHECK IF GRAD NEED TO BE DEACTIVATED!!!
        N_centroids = centroids.shape[0]
        N_batch = x.shape[0]
        dists = (x[:,None,:].repeat(1,N_centroids,1) - centroids[None,:,:].repeat(N_batch,1,1)).norm(dim=2)
        idx = torch.argmin(dists, dim=1)
        return idx, centroids[idx]
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
class SOM_EMB_no_grad(nn.Module):
    # Straight through estimator for centroid assignment in SOM embedding space
    def __init__(self):
        super().__init__()
    def forward(self, x, centroids):
        # x: encoded data of size N x D
        # centroids: SOM dictionary of embeddings  size_som**2 * D
        k, centroid = SOM_EMB_FUNC.apply(x, centroids)
        return x




class BaseModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # input params
        self.batchsize = config['batchsize']
        self.seq_len = config['seq_len']
        self.n_feat = config['n_feat']
        self.n_feat_med = config['n_feat_med']
        self.input_len = int(self.seq_len // 2)
        self.pred_len = int(self.seq_len // 2)
        # model params
        try:
            self.encoder_type = config['encoder_type']
            self.n_emb = config['n_emb']
            self.n_emb_info = config['n_emb_info']
            self.n_emb_med = config['n_emb_med']
        except:
            pass
        self.dropout = config['dropout']
        # SOM params
        try:
            self.som_size = config['som_size']
            self.r_neighbor = config['r_neighbor']
            self.centroids = torch.rand(self.som_size**2, self.n_emb)
            self.som_map, self.som_map_invert = index_map(self.som_size)
        except:
            pass

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
                                   kernel_size=cnn_params['cnn_kernel1'], stride=cnn_params['cnn_stride1'], groups=self.n_feat),
                nn.Sigmoid(),
            )
            # self.encoder_physio = nn.Sequential(
            #     nn.Conv1d(in_channels=7, out_channels=56, kernel_size=6, stride=1, groups=7),
            #     nn.ReLU(),
            #     nn.MaxPool1d(kernel_size=3),
            # )
            self.encoder_med = nn.Sequential(
                nn.Conv1d(in_channels=self.n_feat_med, out_channels=self.n_feat_med * 5,
                          kernel_size=cnn_params['cnn_kernel1'], stride=cnn_params['cnn_stride1'], groups=self.n_feat_med),
                nn.ReLU(),
                nn.Conv1d(in_channels=self.n_feat_med * 5, out_channels=cnn_params['cnn_out2'],
                          kernel_size=cnn_params['cnn_kernel2'], stride=cnn_params['cnn_stride2'], groups=1),
                nn.ReLU(),
                # nn.Conv1d(in_channels=7, out_channels=56, kernel_size=6, stride=1, groups=7),
                # nn.ReLU(),
                # nn.MaxPool1d(kernel_size=3),
                # nn.Conv1d(in_channels=56, out_channels=14, kernel_size=4, stride=1, groups=1),
                # nn.ReLU(),
                # nn.MaxPool1d(kernel_size=2),
            )
            self.encoder_info = nn.Sequential(
                nn.LazyLinear(out_features=self.n_emb_info),
                nn.ReLU()
            )
            self.encoder_physio_addon = nn.Sequential(
                nn.Linear(in_features=self.n_emb + self.n_emb_info, out_features=self.n_emb)
            )

        elif self.encoder_type == 'LSTM':
            pass
        elif self.encoder_type == 'ATT':
            pass
        else:
            pass


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


    def init_som(self):
        self.centroids = torch.normal(mean=0.0, std=0.05,
                                     size=(self.som_size**2, self.n_emb),
                                     requires_grad=True)




class VASO_CLF(BaseModel):
    # Classifier for medication type
    def __init__(self, config):
        super().__init__(config)
        self.beta = config['beta']
        # CNN encoder
        self.init_encoder()
        # attention: patient info <==> vital signs
        self.attention = nn.MultiheadAttention(
            embed_dim=self.n_emb_info, kdim=self.n_emb, vdim=self.n_emb,
            batch_first=True, num_heads=1,
        )
        # SOM layer
        self.init_som()
        self.som_layer = SOM_LAYER()
        # Medication classifier
        self.med_clf = nn.Sequential(
            nn.LazyLinear(out_features=self.n_emb//2),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb//2, out_features=3),
        )

        self.loss_clf = nn.BCEWithLogitsLoss()
        self.loss_weight_constrait = nn.MSELoss()

    def forward(self, data):
        age = data['info']['age']
        height = data['info']['height']
        sex = data['info']['sex']
        apache = data['info']['apache']
        x_info = torch.cat((age, height, sex, apache), dim=1)
        x = data['data'][:, :self.input_len, :].permute(0, 2, 1).float()

        # print(f"000 X: {x.shape}")
        # x_med = data['med'][:, :self.input_len, :].permute(0, 2, 1).double()

        # Encode time-seires vital signs
        x_enc = self.encoder_ae(x)
        # print(f"111 X_ENC: {x_enc.shape}")
        x_hat = self.decoder_ae(x_enc)
        # # print(f"222 X_DEC: {x_hat.shape}")

        # Combine patient info with vital signs
        x_info_enc = self.encoder_info(x_info)
        x_enc = self.encoder_physio_addon(torch.concat((x_enc.reshape(x_enc.shape[0], -1), x_info_enc), dim=1))
        # print(f"222 INFO+PHYSIO OUTPUT: {x_enc.shape}")

        # encode planned medication
        med_pred = self.med_clf(x_enc)
        # print(f"333 CLF OUTPUT: {med_pred.shape}")



        return med_pred, x_hat.permute(0, 2, 1)


    def training_step(self, batch, batch_idx):
        x = batch['data']
        med = batch['med_label'].float()
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        med_, x_hat = self.forward(batch)

        loss_clf = self.loss_clf(med_, med)
        loss_ae = mae_loss(x_hat, x_curr)

        loss = self.beta * loss_ae + loss_clf

        # loss_similarity_enc = self.loss_weight_constrait(self.encoder_ae.layer[0].weight, self.encoder_physio.layer[0].weight)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_clf], ['loss_reconst', 'loss_clf']):
            self.log("train_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        # for metric in self.METRICS:
        #     self.log("train_" + metric, self.METRICS[metric](x_, x[:, :90, :]), on_step=False, on_epoch=True, prog_bar=True,
        #              logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        med = batch['med_label'].float()
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        med_, x_hat = self.forward(batch)

        loss_clf = self.loss_clf(med_, med)
        loss_ae = mae_loss(x_hat, x_curr)

        loss = self.beta * loss_ae + loss_clf

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_clf], ['loss_reconst', 'loss_clf']):
            self.log("val_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        # for metric in self.METRICS:
        #     outputs["val_" + metric] = self.METRICS[metric](x_, x[:, :90, :])
        #     self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
        #              logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']
        med = batch['med_label'].float()
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        med_, x_hat = self.forward(batch)

        loss_clf = self.loss_clf(med_, med)
        loss_ae = mae_loss(x_hat, x_curr)

        loss = self.beta * loss_ae + loss_clf

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_clf], ['loss_reconst', 'loss_clf']):
            self.log("test_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        # for metric in self.METRICS:
        #     outputs["test_" + metric] = self.METRICS[metric](x_, x[:, :90, :])
        #     self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
        #              logger=True)
        return outputs




class VASO_PHYSIO_PRED(BaseModel):
    # Predictor for physio trend when taking medication
    def __init__(self, config):
        super().__init__(config)
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.init_encoder()
        # self.attention = nn.MultiheadAttention(
        #     embed_dim=self.n_emb + self.n_emb_info, dropout=0, batch_first=True
        # )

        self.decoder_pred = nn.Sequential(
            nn.ConvTranspose1d(in_channels=14, out_channels=56, kernel_size=3, stride=2, groups=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=56, out_channels=7, kernel_size=6, stride=3, groups=7),
            nn.Sigmoid(),
        )
        self.state_transition = nn.Sequential(
            nn.Linear(in_features=self.n_emb+self.n_emb_info, out_features=self.n_emb),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb, out_features=self.n_emb),
        )
        self.control_input = nn.Sequential(
            nn.Linear(in_features=self.n_emb + self.n_emb_info, out_features=self.n_emb),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb, out_features=self.n_emb),
        )

        self.loss_pred = mae_loss
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

        # print(f"000 X: {x.shape} --- MED: {med.shape}")
        # x_med = data['med'][:, :self.input_len, :].permute(0, 2, 1).double()

        # Encode time-seires vital signs
        x_enc = self.encoder_ae(x)
        # print(f"111 X_ENC: {x_enc.shape}")
        x_hat = self.decoder_ae(x_enc)
        # print(f"222 X_DEC: {x_hat.shape}")

        # Combine patient info with vital signs
        x_info_enc = self.encoder_info(x_info)
        x_enc = self.encoder_physio_addon(torch.concat((x_enc.reshape(x_enc.shape[0], -1), x_info_enc), dim=1))
        # print(f"222 INFO+PHYSIO ENC: {x_enc.shape}")

        # encode planned medication
        med_enc = self.encoder_med(med)
        # print(f"333 MED ENC: {med_enc.shape}")

        # mimic Kalman filter
        x_enc_next = self.state_transition(torch.concat((x_enc.reshape(x_enc.shape[0], -1), x_info_enc), dim=1)) \
                     + self.control_input(torch.concat((med_enc.reshape(x_enc.shape[0], -1), x_info_enc), dim=1))
        # print(f"444 X ENC NEXT: {x_enc_next.shape}")

        # decode patient state in to time-series vital signs
        x_next_hat = self.decoder_pred(x_enc_next.reshape(med_enc.shape))
        # print(f"555 X PRED: {x_next_hat.shape}")

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

        loss = self.alpha * loss_pred + self.beta * loss_ae + loss_similarity_dec

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_pred, loss_similarity_dec], ['loss_reconst', 'loss_pred', 'loss_constrain']):
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

        loss = self.alpha * loss_pred + self.beta * loss_ae + loss_similarity_dec

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_pred, loss_similarity_dec],
                                    ['loss_reconst', 'loss_pred', 'loss_constrain']):
            self.log("test_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return outputs




class VASO_DOSAGE_PRED(BaseModel):
    # Predictor for physio trend when taking medication
    def __init__(self, config):
        super().__init__(config)
        self.rnn_hidden = config['rnn_hidden']
        self.load_ae_physio(config['ae_model_path'])

        self.lstm = nn.LSTM(input_size=self.n_feat, hidden_size=self.rnn_hidden, num_layers=1,
                                    dropout=self.dropout, batch_first=True)
        self.dosage_clf = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=self.rnn_hidden, out_features=3),
            nn.Sigmoid()
        )

        self.loss = TemporalCrossEntropy(num_classes=3)

    def load_ae_physio(self, model_path):
        files = os.listdir(model_path)
        model_file = None
        model_config = None
        for file in files:
            if "val_loss_pred" in file:
                model_file = os.path.join(model_path, file)
            elif file == "model_config.p":
                model_config = pickle_load(os.path.join(model_path, file))
        self.ae_physio = VASO_PHYSIO_PRED.load_from_checkpoint(checkpoint_path=model_file, config=model_config)
        self.ae_physio.to(self.device)
        self.ae_physio.eval()


    def forward(self, data):
        # age = data['info']['age']
        # height = data['info']['height']
        # sex = data['info']['sex']
        # apache = data['info']['apache']
        # x_info = torch.cat((age, height, sex, apache), dim=1)
        # x = data['data'][:, :self.input_len, :].permute(0, 2, 1).float()
        # med = data['med'][:, self.input_len:, :self.n_feat_med].permute(0, 2, 1).float()
        # med[torch.isnan(med)] = 0

        _, x_next_hat = self.ae_physio(data)
        batchsize = x_next_hat.size(0)
        device = x_next_hat.device

        # print(f"111 X_PRED: {x_next_hat.shape}, X_NEXT: {data['data'][:, self.input_len:, :].shape}")
        x_diff = x_next_hat - data['data'][:, self.input_len:, :]

        h0 = torch.zeros(1, batchsize, self.rnn_hidden).to(device)
        c0 = torch.zeros(1, batchsize, self.rnn_hidden).to(device)
        out, (h, c) = self.lstm(x_diff.float(), (h0, c0))     # out: (batch_size, seq_length, hidden_size)

        dosage_pred = torch.empty((batchsize, 0, 3)).to(device)
        for i in range(out.shape[1]):
            pred = self.dosage_clf(out[:,i,:])
            dosage_pred = torch.concat((dosage_pred, pred[:,None,:]), dim=1)
        # print(f"222 DOSAGE_TREND: {dosage_pred.shape}")

        return dosage_pred


    def training_step(self, batch, batch_idx):
        dosage_trend_gt = batch['dosage_trend_bool'][:,self.input_len:,:]
        dosage_trend = self.forward(batch)

        loss = self.loss(dosage_trend, dosage_trend_gt)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        dosage_trend_gt = batch['dosage_trend_bool'][:,self.input_len:,:]
        dosage_trend = self.forward(batch)

        loss = self.loss(dosage_trend, dosage_trend_gt)

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return outputs

    def test_step(self, batch, batch_idx):
        dosage_trend_gt = batch['dosage_trend_bool'][:,self.input_len:,:]
        dosage_trend = self.forward(batch)

        loss = self.loss(dosage_trend, dosage_trend_gt)

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return outputs


class VASO_PHYSIO_PRED_MONO(BaseModel):
    # Predictor for physio trend when taking medication
    def __init__(self, config):
        super().__init__(config)
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.n_emb_mono = config['n_emb_mono']
        self.n_groupsort = config['n_groupsort']
        self.monotonic_constraints = config['monotonic_constraints']
        self.init_encoder()
        # self.attention = nn.MultiheadAttention(
        #     embed_dim=self.n_emb + self.n_emb_info, dropout=0, batch_first=True
        # )

        self.state_transition = nn.Sequential(
            nn.Linear(in_features=self.n_emb+self.n_emb_info, out_features=self.n_emb),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb, out_features=self.n_emb),
        )

        self.monotonic_layer = lmn.MonotonicWrapper(
            nn.Sequential(
                lmn.LipschitzLinear(self.n_feat_med + self.n_emb, self.n_emb_mono, kind="one-inf"),
                lmn.GroupSort(self.n_groupsort),
                lmn.LipschitzLinear(self.n_emb_mono, self.n_feat, kind="inf"),
            ),
            monotonic_constraints=self.monotonic_constraints.astype(np.double)
        )
        #TODO: REDUCE N_EMB, SET RECURRENT PREDICTION IN FORWARD FUNCTION

        # lmn.MonotonicLayer(2, 3, monotonic_constraints=[[1, 0, -1], [0, 1, 0]])

        # lip_nn = nn.Sequential(
        #     lmn.LipschitzLinear(2, 32, kind="one-inf"),
        #     lmn.GroupSort(2),
        #     lmn.LipschitzLinear(32, 2, kind="inf"),
        # )
        # monotonic_nn = lmn.MonotonicWrapper(lip_nn, monotonic_constraints=[1, 0])

        self.loss_pred = mae_loss



    def forward(self, data):
        age = data['info']['age']
        height = data['info']['height']
        sex = data['info']['sex']
        apache = data['info']['apache']
        x_info = torch.cat((age, height, sex, apache), dim=1)
        x = data['data'][:, :self.input_len, :].permute(0, 2, 1).float()
        # med = data['med'][:, self.input_len:, :].permute(0, 2, 1).float()
        # med[torch.isnan(med)] = 0

        # print(f"000 X: {x.shape} --- MED: {med.shape}")
        x_med = data['med'][:, :self.input_len, :].permute(0, 2, 1).float()
        x_med[torch.isnan(x_med)] = 0
        # print(f"000 X: {x.shape} --- X_MED: {x_med.shape}")
        # Encode time-seires vital signs
        x_enc = self.encoder_ae(x)
        # print(f"111 X_ENC: {x_enc.shape}")
        x_hat = self.decoder_ae(x_enc)
        # print(f"222 X_DEC: {x_hat.shape}")

        # Combine patient info with vital signs
        x_info_enc = self.encoder_info(x_info)
        # print(f"222 INFO+PHYSIO ENC: {x_enc.shape}")
        x_enc_next = self.state_transition(torch.concat((x_enc.reshape(x_enc.shape[0], -1), x_info_enc), dim=1))


        # Predict future vital sign with monotonic constraints on medication
        # print(f"CHECK: {x_enc_next.reshape(x_enc_next.shape[0], -1).shape}   {x_med[:,:,0].shape}   {torch.concat((x_enc_next.reshape(x_enc_next.shape[0], -1), x_med[:,:,0]), dim=1).shape}")
        x_next_hat = self.monotonic_layer(
            torch.concat((x_enc_next.reshape(x_enc_next.shape[0], -1), x_med[:, :, 0]), dim=1)
        )
        x_next_hat = x_next_hat[:,:,None]
        # print(f"333 INFO X_NEXT_HAT: {x_next_hat.shape}")
        for i in range(1, self.pred_len):
            if i == 0:
                x_enc_ = x_enc_next.copy()
            else:
                x_curr = torch.concat((x[:, :, i:], x_next_hat), dim=2)
                x_enc = self.encoder_ae(x_curr)
                x_enc_ = self.state_transition(torch.concat((x_enc.reshape(x_enc.shape[0], -1), x_info_enc), dim=1))

            x_next = self.monotonic_layer(
                torch.concat((x_enc_.reshape(x_enc_.shape[0], -1), x_med[:,:,i]), dim=1)
            )
            x_next_hat = torch.concat((x_next_hat, x_next[:, :, None]), dim=2)
        # print(f"333 INFO X_NEXT_HAT: {x_next_hat.shape}")

        return x_hat.permute(0, 2, 1), x_next_hat.permute(0, 2, 1)


    def training_step(self, batch, batch_idx):
        x = batch['data']
        x_mask = batch['data_mask']
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_hat, x_next_hat = self.forward(batch)
        # print(torch.isnan(x.reshape(-1)).sum().item(),
        #       torch.isnan(x_hat.reshape(-1)).sum().item(),
        #       torch.isnan(x_next_hat.reshape(-1)).sum().item())

        loss_ae = mae_loss(x_hat, x_curr, x_mask[:, :90, :])
        loss_pred = mae_loss(x_next_hat, x_next, x_mask[:, 90:, :])

        loss = self.alpha * loss_pred + self.beta * loss_ae

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_pred], ['loss_reconst', 'loss_pred', 'loss_constrain']):
            self.log("train_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        x_mask = batch['data_mask']
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_hat, x_next_hat = self.forward(batch)

        loss_ae = mae_loss(x_hat, x_curr, x_mask[:, :90, :])
        loss_pred = mae_loss(x_next_hat, x_next, x_mask[:, 90:, :])

        loss = self.alpha * loss_pred + self.beta * loss_ae

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_pred],
                                    ['loss_reconst', 'loss_pred', 'loss_constrain']):
            self.log("val_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']
        x_mask = batch['data_mask']
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_hat, x_next_hat = self.forward(batch)

        loss_ae = mae_loss(x_hat, x_curr, x_mask[:, :90, :])
        loss_pred = mae_loss(x_next_hat, x_next, x_mask[:, 90:, :])

        loss = self.alpha * loss_pred + self.beta * loss_ae

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_pred],
                                    ['loss_reconst', 'loss_pred', 'loss_constrain']):
            self.log("test_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return outputs


class TemporalCrossEntropy(nn.Module):
    def __init__(self, num_classes=3, weight=[.1, 1., 1.]):
        """
        Loss function for temporal sequence classification with a tolerance window.
        :param num_classes: Integer, total number of classes (time steps).
        """
        super().__init__()
        self.tolerance_window = 2
        self.window = torch.Tensor([.25, .5, 1, .5, .25])
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean', weight=torch.tensor(weight))

    def forward(self, probs, targets):
        """
        Compute the temporal tolerance loss.
        :param probs: Tensor, shape (batch_size, seq_length, num_classes), raw model predictions.
        :param targets: Tensor, shape (batch_size, seq_length, num_classes), true time step labels.
        :return: Tensor, scalar loss value.
        """
        batch_size, seq_length, num_classes = probs.shape

        targets_ = torch.empty(0, seq_length, num_classes).to(probs.device)
        for b in range(batch_size):
            targets_tol = torch.empty(seq_length, 0).to(probs.device)
            for c in range(num_classes):
                indices = torch.nonzero(targets[b,:,c]==1)
                if len(indices) == 0:
                    targets_c = torch.zeros(seq_length).to(probs.device)
                else:
                    targets_c = torch.empty(0, seq_length).to(probs.device)
                    for idx in indices:
                        start = max(idx-2, 0)
                        end = min(idx+2+1, seq_length)
                        t = torch.zeros(1, seq_length).to(probs.device)
                        try:
                            t[0, start:end] = self.window[(start-idx+2):(end-idx+2)]
                        except:
                            print(idx)
                        targets_c = torch.concat((targets_c, t), dim=0)
                    targets_c, _ = torch.max(targets_c, dim=0)
                targets_tol = torch.concat((targets_tol, targets_c[:, None]), dim=1)
            targets_ = torch.concat((targets_, targets_tol[None, :, :]), dim=0)

        ce_loss = self.ce_loss(probs.reshape(-1, num_classes), targets_.reshape(-1, num_classes),)

        return ce_loss



# class SOM_CLF(BaseModel):
#     # Classifier for medication type
#     def __init__(self, config):
#         super().__init__(config)
#         self.beta = config['beta']
#         # CNN encoder
#         self.init_encoder()
#         # attention: patient info <==> vital signs
#         self.attention = nn.MultiheadAttention(
#             embed_dim=self.n_emb_info, kdim=self.n_emb, vdim=self.n_emb,
#             batch_first=True, num_heads=1,
#         )
#         # SOM layer
#         self.init_som()
#         self.som_layer = SOM_LAYER()
#         # Medication classifier
#         self.med_clf = nn.Sequential(
#             nn.LazyLinear(out_features=self.n_emb//2),
#             nn.ReLU(),
#             nn.Linear(in_features=self.n_emb//2, out_features=3),
#         )
#
#         self.loss_clf = nn.BCEWithLogitsLoss()
#         self.loss_weight_constrait = nn.MSELoss()
#
#     def forward(self, data):
#         age = data['info']['age']
#         height = data['info']['height']
#         sex = data['info']['sex']
#         apache = data['info']['apache']
#         x_info = torch.cat((age, height, sex, apache), dim=1)
#         x = data['data'][:, :self.input_len, :].permute(0, 2, 1).float()
#
#         # print(f"000 X: {x.shape}")
#         # x_med = data['med'][:, :self.input_len, :].permute(0, 2, 1).double()
#
#         # Encode time-seires vital signs
#         x_enc = self.encoder_ae(x)
#         # print(f"111 X_ENC: {x_enc.shape}")
#         x_hat = self.decoder_ae(x_enc)
#         # # print(f"222 X_DEC: {x_hat.shape}")
#
#         # Combine patient info with vital signs
#         x_info_enc = self.encoder_info(x_info)
#         x_enc = self.encoder_physio_addon(torch.concat((x_enc.reshape(x_enc.shape[0], -1), x_info_enc), dim=1))
#         # print(f"222 INFO+PHYSIO OUTPUT: {x_enc.shape}")
#
#         # encode planned medication
#         med_pred = self.med_clf(x_enc)
#         # print(f"333 CLF OUTPUT: {med_pred.shape}")
#
#
#
#         return med_pred, x_hat.permute(0, 2, 1)
#
#
#     def training_step(self, batch, batch_idx):
#         x = batch['data']
#         med = batch['med_label'].float()
#         x_curr = x[:, :self.input_len, :]
#         x_next = x[:, self.input_len:, :]
#         med_, x_hat = self.forward(batch)
#
#         loss_clf = self.loss_clf(med_, med)
#         loss_ae = mae_loss(x_hat, x_curr)
#
#         loss = self.beta * loss_ae + loss_clf
#
#         # loss_similarity_enc = self.loss_weight_constrait(self.encoder_ae.layer[0].weight, self.encoder_physio.layer[0].weight)
#
#         self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         for loss_, loss_type in zip([loss_ae, loss_clf], ['loss_reconst', 'loss_clf']):
#             self.log("train_" + loss_type, loss_, on_step=False, on_epoch=True,
#                      prog_bar=True, logger=True)
#         # for metric in self.METRICS:
#         #     self.log("train_" + metric, self.METRICS[metric](x_, x[:, :90, :]), on_step=False, on_epoch=True, prog_bar=True,
#         #              logger=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         x = batch['data']
#         med = batch['med_label'].float()
#         x_curr = x[:, :self.input_len, :]
#         x_next = x[:, self.input_len:, :]
#         med_, x_hat = self.forward(batch)
#
#         loss_clf = self.loss_clf(med_, med)
#         loss_ae = mae_loss(x_hat, x_curr)
#
#         loss = self.beta * loss_ae + loss_clf
#
#         outputs = {'val_loss': loss}
#         self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         for loss_, loss_type in zip([loss_ae, loss_clf], ['loss_reconst', 'loss_clf']):
#             self.log("val_" + loss_type, loss_, on_step=False, on_epoch=True,
#                      prog_bar=True, logger=True)
#         # for metric in self.METRICS:
#         #     outputs["val_" + metric] = self.METRICS[metric](x_, x[:, :90, :])
#         #     self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
#         #              logger=True)
#         return outputs
#
#     def test_step(self, batch, batch_idx):
#         x = batch['data']
#         med = batch['med_label'].float()
#         x_curr = x[:, :self.input_len, :]
#         x_next = x[:, self.input_len:, :]
#         med_, x_hat = self.forward(batch)
#
#         loss_clf = self.loss_clf(med_, med)
#         loss_ae = mae_loss(x_hat, x_curr)
#
#         loss = self.beta * loss_ae + loss_clf
#
#         outputs = {'test_loss': loss}
#         self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         for loss_, loss_type in zip([loss_ae, loss_clf], ['loss_reconst', 'loss_clf']):
#             self.log("test_" + loss_type, loss_, on_step=False, on_epoch=True,
#                      prog_bar=True, logger=True)
#         # for metric in self.METRICS:
#         #     outputs["test_" + metric] = self.METRICS[metric](x_, x[:, :90, :])
#         #     self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
#         #              logger=True)
#         return outputs




class SOM_PRED(BaseModel):
    # Predictor for physio trend when taking medication
    def __init__(self, config):
        super().__init__(config)
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.init_encoder()
        self.init_som()
        # self.attention = nn.MultiheadAttention(
        #     embed_dim=self.n_emb + self.n_emb_info, dropout=0, batch_first=True
        # )

        self.decoder_pred = nn.Sequential(
            nn.ConvTranspose1d(in_channels=14, out_channels=56, kernel_size=3, stride=2, groups=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=56, out_channels=7, kernel_size=6, stride=3, groups=7),
            nn.Sigmoid(),
        )
        self.state_transition = nn.Sequential(
            nn.Linear(in_features=self.n_emb+self.n_emb_info, out_features=self.n_emb),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb, out_features=self.n_emb),
        )
        self.control_input = nn.Sequential(
            nn.Linear(in_features=self.n_emb + self.n_emb_info, out_features=self.n_emb),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb, out_features=self.n_emb),
        )

        self.loss_pred = mae_loss
        self.loss_weight_constrait = nn.MSELoss()


    def forward(self, data):
        age = data['info']['age']
        height = data['info']['height']
        sex = data['info']['sex']
        apache = data['info']['apache']
        x_info = torch.cat((age, height, sex, apache), dim=1)
        x = data['data'][:, :self.input_len, :].permute(0, 2, 1).float()
        med = data['med'][:, self.input_len:, :].permute(0, 2, 1).float()
        med[torch.isnan(med)] = 0

        # print(f"000 X: {x.shape} --- MED: {med.shape}")
        # x_med = data['med'][:, :self.input_len, :].permute(0, 2, 1).double()

        # Encode time-seires vital signs
        x_enc = self.encoder_ae(x)
        # print(f"111 X_ENC: {x_enc.shape}")
        x_hat = self.decoder_ae(x_enc)
        # print(f"222 X_DEC: {x_hat.shape}")

        # Combine patient info with vital signs
        x_info_enc = self.encoder_info(x_info)
        x_enc = self.encoder_physio_addon(torch.concat((x_enc.reshape(x_enc.shape[0], -1), x_info_enc), dim=1))
        # print(f"222 INFO+PHYSIO ENC: {x_enc.shape}")

        # encode planned medication
        med_enc = self.encoder_med(med)
        # print(f"333 MED ENC: {med_enc.shape}")

        # mimic Kalman filter
        x_enc_next = self.state_transition(torch.concat((x_enc.reshape(x_enc.shape[0], -1), x_info_enc), dim=1)) \
                     + self.control_input(torch.concat((med_enc.reshape(x_enc.shape[0], -1), x_info_enc), dim=1))
        # print(f"444 X ENC NEXT: {x_enc_next.shape}")

        # decode patient state in to time-series vital signs
        x_next_hat = self.decoder_pred(x_enc_next.reshape(med_enc.shape))
        # print(f"555 X PRED: {x_next_hat.shape}")

        return x_hat.permute(0, 2, 1), x_next_hat.permute(0, 2, 1)

    def z_e(self, x):
        ###TODO: CHECK IF GRAD NEED TO BE DEACTIVATED!!!
        N_centroids = self.som_size ** 2
        N_batch = self.batchsize
        dists = (x[:,None,:].repeat(1,N_centroids,1) - self.centroids[None,:,:].repeat(N_batch,1,1)).norm(dim=2)
        k = torch.argmin(dists, dim=1)
        return k, self.centroids[k]

    def z_neighbors(self, k):
        kx = k // self.som_size
        ky = k % self.som_size
        ky_not_top = torch.where(ky<self.som_size-1)
        ky_not_bottom = torch.where(ky>0)
        kx_not_left = torch.where(kx>0)
        kx_not_right = torch.where(kx<self.som_size-1)

        k_top = torch.concat((kx[ky_not_top][None, :, None], ky[ky_not_top][None, :, None] + 1), axis=2)
        k_bottom = torch.concat((kx[ky_not_bottom][None, :, None], ky[ky_not_bottom][None, :, None] - 1), axis=2)
        k_left = torch.concat((kx[kx_not_left][None, :, None] - 1, ky[kx_not_left][None, :, None]), axis=2)
        k_right = torch.concat((kx[kx_not_right][None, :, None] + 1, ky[kx_not_right][None, :, None]), axis=2)

        k_neighbors = torch.concat((k_top, k_))


    def training_step(self, batch, batch_idx):
        x = batch['data']
        x_mask = batch['data_mask']
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_hat, x_next_hat = self.forward(batch)
        # print(torch.isnan(x.reshape(-1)).sum().item(),
        #       torch.isnan(x_hat.reshape(-1)).sum().item(),
        #       torch.isnan(x_next_hat.reshape(-1)).sum().item())

        loss_ae = mae_loss(x_hat, x_curr, x_mask[:, :90, :])
        loss_pred = mae_loss(x_next_hat, x_next, x_mask[:, 90:, :])
        loss_similarity_dec = self.loss_weight_constrait(self.decoder_ae[0].weight,
                                                         self.decoder_pred[0].weight) \
                              + self.loss_weight_constrait(self.decoder_ae[2].weight,
                                                           self.decoder_pred[2].weight)

        loss = self.alpha * loss_pred + self.beta * loss_ae  + loss_similarity_dec

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_pred, loss_similarity_dec], ['loss_reconst', 'loss_pred', 'loss_constrain']):
            self.log("train_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        x_mask = batch['data_mask']
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_hat, x_next_hat = self.forward(batch)

        loss_ae = mae_loss(x_hat, x_curr, x_mask[:, :90, :])
        loss_pred = mae_loss(x_next_hat, x_next, x_mask[:, 90:, :])
        loss_similarity_dec = self.loss_weight_constrait(self.decoder_ae[0].weight,
                                                         self.decoder_pred[0].weight) \
                              + self.loss_weight_constrait(self.decoder_ae[2].weight,
                                                           self.decoder_pred[2].weight)

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
        x_hat, x_next_hat = self.forward(batch)

        loss_ae = mae_loss(x_hat, x_curr, x_mask[:, :90, :])
        loss_pred = mae_loss(x_next_hat, x_next, x_mask[:, 90:, :])
        loss_similarity_dec = self.loss_weight_constrait(self.decoder_ae[0].weight,
                                                         self.decoder_pred[0].weight) \
                              + self.loss_weight_constrait(self.decoder_ae[2].weight,
                                                           self.decoder_pred[2].weight)

        loss = self.alpha * loss_pred + self.beta * loss_ae + loss_similarity_dec

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for loss_, loss_type in zip([loss_ae, loss_pred, loss_similarity_dec],
                                    ['loss_reconst', 'loss_pred', 'loss_constrain']):
            self.log("test_" + loss_type, loss_, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        return outputs



# class SOM_MTL(BaseModel):
#     # Multi-task learning for both CLS and PRED
#     def __init__(self):
#         self.init_encoder()
#         self.attention = nn.MultiheadAttention(
#             embed_dim=self.n_emb + self.n_emb_info, dropout=0, batch_first=True
#         )
#         self.med_clf = nn.Sequential(
#             nn.Linear(in_features=self.n_emb, out_features=self.n_emb // 2),
#             nn.ReLU(),
#             nn.Linear(in_features=self.n_emb // 2, out_features=7),
#         )
#         self.clf_loss = nn.CrossEntropyLoss()