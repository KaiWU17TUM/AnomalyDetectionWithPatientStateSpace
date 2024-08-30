import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torch.autograd as autograd

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


med_dict = {
    0: [1,0,0],
    1: [0,1,0],
    2: [0,0,1],
}

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


class SOM_EMB_FUNC(autograd.Function):
    @staticmethod
    def forward(ctx, x, centroids):
        ###TODO: CHECK IF GRAD NEED TO BE DEACTIVATED!!!
        N_centroids = centroids.shape[0]
        N_batch = x.shape[0]
        dists = (x[:,None,:].repeat(1,N_centroids,1) - centroids[None,:,:].repeat(N_batch,1,1)).norm(dim=2)
        idx = torch.argmin(dists, dim=1)
        return centroids[idx]
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
class SOM_LAYER(nn.Module):
    # Straight through estimator for centroid assignment in SOM embedding space
    def __init__(self):
        super().__init__()
    def forward(self, x, centoids):
        # x: encoded data of size N x D
        # centroids: SOM dictionary of embeddings  size_som**2 * D
        x = SOM_EMB_FUNC.apply(x, centoids)
        return x


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
        self.n_emb_info = config['n_emb_info']
        self.n_emb_med = config['n_emb_med']
        self.dropout = config['dropout']
        # SOM params
        self.som_size = config['som_size']
        self.r_neighbor = config['r_neighbor']
        self.emb_dict = torch.rand(self.som_size**2, self.n_emb)
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
            self.encoder_ae = nn.Sequential(
                nn.Conv1d(in_channels=7, out_channels=56, kernel_size=6, stride=3, groups=7),
                nn.ReLU(),
                nn.Conv1d(in_channels=56, out_channels=14, kernel_size=3, stride=2, groups=1),
                nn.ReLU(),
            )
            self.decoder_ae = nn.Sequential(
                nn.ConvTranspose1d(in_channels=14, out_channels=56, kernel_size=3, stride=2, groups=1),
                nn.ReLU(),
                nn.ConvTranspose1d(in_channels=56, out_channels=7, kernel_size=6, stride=3, groups=7),
                nn.Sigmoid(),
            )
            # self.encoder_physio = nn.Sequential(
            #     nn.Conv1d(in_channels=7, out_channels=56, kernel_size=6, stride=1, groups=7),
            #     nn.ReLU(),
            #     nn.MaxPool1d(kernel_size=3),
            # )
            self.encoder_med = nn.Sequential(
                nn.Conv1d(in_channels=7, out_channels=56, kernel_size=6, stride=3, groups=7),
                nn.ReLU(),
                nn.Conv1d(in_channels=56, out_channels=14, kernel_size=3, stride=2, groups=1),
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

    def init_som(self):
        self.emb_dict = torch.normal(mean=0.0, std=0.05,
                                     size=(self.som_size**2, self.n_emb),
                                     requires_grad=True)






class SOM_CLF(BaseModelSOM):
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




class SOM_PRED(BaseModelSOM):
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


    def training_step(self, batch, batch_idx):
        x = batch['data']
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_hat, x_next_hat = self.forward(batch)
        # print(torch.isnan(x.reshape(-1)).sum().item(),
        #       torch.isnan(x_hat.reshape(-1)).sum().item(),
        #       torch.isnan(x_next_hat.reshape(-1)).sum().item())

        loss_ae = mae_loss(x_hat, x_curr)
        loss_pred = mae_loss(x_next_hat, x_next)
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
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_hat, x_next_hat = self.forward(batch)

        loss_ae = mae_loss(x_hat, x_curr)
        loss_pred = mae_loss(x_next_hat, x_next)
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
        x_curr = x[:, :self.input_len, :]
        x_next = x[:, self.input_len:, :]
        x_hat, x_next_hat = self.forward(batch)

        loss_ae = mae_loss(x_hat, x_curr)
        loss_pred = mae_loss(x_next_hat, x_next)
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




class SOM_MTL(BaseModelSOM):
    # Multi-task learning for both CLS and PRED
    def __init__(self):
        self.init_encoder()
        self.attention = nn.MultiheadAttention(
            embed_dim=self.n_emb + self.n_emb_info, dropout=0, batch_first=True
        )
        self.med_clf = nn.Sequential(
            nn.Linear(in_features=self.n_emb, out_features=self.n_emb // 2),
            nn.ReLU(),
            nn.Linear(in_features=self.n_emb // 2, out_features=7),
        )
        self.clf_loss = nn.CrossEntropyLoss()