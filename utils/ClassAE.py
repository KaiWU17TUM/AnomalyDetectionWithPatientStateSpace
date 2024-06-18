import numpy as np
from collections import OrderedDict

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

# from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerConfig


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

class LSTMEncoder(nn.Module):
    def __init__(self, config, n_feat=10, n_emb=128, n_layer=2, dropout=0):
        super().__init__()
        self.config = config
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
        batchsize = x.shape[0]

        h0 = torch.zeros(self.n_layer, batchsize, self.n_emb, device=self.config['device'])
        c0 = torch.zeros(self.n_layer, batchsize, self.n_emb, device=self.config['device'])

        # print(f'LSTM ENCODER INPUT: {x.shape}')
        out, hn = self.lstm(x.float(), (h0, c0))
        # print(f'LSTM ENCODER INPUT: {out.shape}')

        return out, hn


class LSTMDecoder_recurrent(nn.Module):
    def __init__(self, config, seq_len=181, n_feat=12, n_emb=128, n_layer=2, dropout=0):
        super().__init__()
        self.config = config
        self.seq_len = seq_len
        self.n_feat = n_feat
        self.n_emb = n_emb
        self.n_layer = n_layer
        self.dropout = dropout


        self.lstm = nn.LSTM(
            input_size=self.n_emb,
            hidden_size=self.n_emb,
            num_layers=self.n_layer,
            dropout=self.dropout,
            batch_first=True,
        )
        self.out = nn.Linear(in_features=self.n_layer*self.n_emb, out_features=self.n_feat)


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


class LSTMDecoder(nn.Module):
    def __init__(self, config, seq_len=181, n_feat=12, n_emb=128, n_layer=2, dropout=0):
        super().__init__()
        self.config = config
        self.seq_len = seq_len
        self.n_feat = n_feat
        self.n_emb = n_emb
        self.n_layer = n_layer
        self.dropout = dropout


        self.lstm = nn.LSTM(
            input_size=self.n_emb,
            hidden_size=self.n_emb,
            num_layers=self.n_layer,
            dropout=self.dropout,
            batch_first=True,
        )
        self.out = nn.Linear(in_features=self.n_emb, out_features=self.n_feat)


    def forward(self, input):
        h0, c0 = input
        batchsize = h0.shape[1]

        if self.n_layer > 1:
            x = h0[-1,:,:].reshape(batchsize, -1)

        # print(f'DECODER LSTM INPUT: {x.shape}')
        x = x[:, None, :].repeat(1, self.seq_len, 1)
        # print(f'DECODER LSTM INPUT: {x.shape}')
        out, _ = self.lstm(x)
        out = F.relu(out)
        # print(f'DECODER LSTM OUTPUT: {out.shape}')
        x_ = self.out(out)
        x_ = F.sigmoid(x_)
        # print(f'DECODER FC OUTPUT: {x_.shape}')

        return x_


class BaseAE(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len = config['seq_len']
        self.n_feat = config['n_feat']
        if 'n_feat_med' in config:
            self.n_feat_med = config['n_feat_med']
        self.n_emb = config['n_emb']
        self.n_layer = config['n_layer']
        self.dropout = config['dropout']
        self.input_len = int(self.seq_len // 2 + 1)
        self.pred_len = int(self.seq_len // 2)

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


class LSTM_AE(BaseAE):
    def __init__(self, config):
        super().__init__(config)

        # self.config = config
        # self.seq_len = config['seq_len']
        # self.n_feat = config['n_feat']
        # self.n_emb = config['n_emb']
        # self.n_layer = config['n_layer']
        # self.dropout = config['dropout']

        self.encoder = LSTMEncoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)
        self.decoder = LSTMDecoder_recurrent(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)

        # self.METRICS = {
        #     'mse': mse_loss,
        # }


    def forward(self, x):
        _, enc_hn = self.encoder(x)
        x_ = self.decoder(enc_hn)

        return x_

    def training_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x[:,:,2:])
        # loss = F.l1_loss(x_, x)
        loss = mae_loss(x_, x)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](x_, x), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x[:,:,2:])
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
        x_ = self.forward(x[:,:,2:])
        # loss = F.l1_loss(x_, x)
        loss = mae_loss(x_, x)

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](x_, x)
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs


class GuidedLSTM_AE(BaseAE):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = LSTMEncoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)
        self.decoder = LSTMDecoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)

        self.att = nn.MultiheadAttention(embed_dim=1, kdim=self.n_feat, vdim=self.n_feat,
                                         num_heads=1, dropout=self.dropout, batch_first=True)
        self.fc_att = nn.Linear(in_features=self.seq_len, out_features= self.seq_len * self.n_feat)
        self.fc_h = nn.Linear(in_features=self.seq_len * self.n_feat + self.n_layer * self.n_emb,
                              out_features=self.n_layer * self.n_emb)
        self.fc_c = nn.Linear(in_features=self.seq_len * self.n_feat + self.n_layer * self.n_emb,
                              out_features=self.n_layer * self.n_emb)

    def forward(self, x):
        batchsize = x.shape[0]
        x = x.float()
        pharma_mask = x[:, :, 0]
        dosage = x[:, :, 1]
        dosage = dosage[:, :, None]
        x = x[:, :, 2:]
        len_pred = x.shape[1] - 1

        _, enc_hn = self.encoder(x)

        attn_output, _ = self.att(dosage, x, x, need_weights=False)
        attn_output = self.fc_att(attn_output.reshape(batchsize, -1))
        attn_output = F.relu(attn_output)
        # print(f'111 hn: {enc_hn[0].shape}, cn: {enc_hn[1].shape}, permute:{enc_hn[0].permute(1,0,2).shape}')
        hn = enc_hn[0].permute(1,0,2).reshape(batchsize, -1)
        cn = enc_hn[1].permute(1,0,2).reshape(batchsize, -1)
        # print(f'222 hn: {hn.shape}, cn:{cn.shape}, attn_output: {attn_output.shape}')
        # print(f'333 FC INPUT: {torch.concat((attn_output.reshape(batchsize, -1), hn), dim=1).shape}')
        hn_ = self.fc_h(torch.concat((attn_output.reshape(batchsize, -1), hn), dim=1))
        cn_ = self.fc_c(torch.concat((attn_output.reshape(batchsize, -1), cn), dim=1))
        hn_ = hn_.reshape(batchsize, self.n_layer, self.n_emb).permute(1, 0, 2).contiguous()
        cn_ = cn_.reshape(batchsize, self.n_layer, self.n_emb).permute(1, 0, 2).contiguous()
        # print(f'444 hn_: {hn_.shape}, cn_:{cn_.shape}')

        x_ = self.decoder((hn_, cn_))

        # print(f'555 output x: {x_.shape}')

        return x_

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


class LSTM_AE_ALLMED(BaseAE):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = LSTMEncoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)
        self.decoder = LSTMDecoder(config=config, seq_len=self.seq_len, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)

    def forward(self, x):
        batchsize = x.shape[0]
        x = x.float()
        x = x[:, :, -10:]

        out, enc_hn = self.encoder(x)
        x_ = self.decoder(out[:, -1, :])

        return x_

    def training_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x[:, :, 10:])
        loss = mae_loss(x_, x[:, :, 10:])

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](x_, x[:, :, 10:]), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x[:, :, 10:])
        # loss = F.l1_loss(x_, x)
        loss = mae_loss(x_, x[:, :, 10:])

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](x_, x[:, :, 10:])
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x[:, :, 10:])
        # loss = F.l1_loss(x_, x)
        loss = mae_loss(x_, x[:, :, 10:])

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](x_, x[:, :, 10:])
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs


class GuidedLSTM_AE_ALLMED(BaseAE):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = LSTMEncoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)
        self.decoder = LSTMDecoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)

        self.att = nn.MultiheadAttention(embed_dim=9, kdim=self.n_feat, vdim=self.n_feat,
                                         num_heads=1, dropout=self.dropout, batch_first=True)

        self.fc_enc = nn.Linear(in_features=self.n_emb, out_features=self.n_emb)
        self.fc_att = nn.Linear(in_features=9 * self.input_len, out_features=self.n_emb)

        self.fc2dec = nn.Linear(in_features=self.n_emb * 2, out_features=self.n_emb)

        # self.fc_h = nn.Linear(in_features=self.seq_len * self.n_feat + self.n_layer * self.n_emb,
        #                       out_features=self.n_layer * self.n_emb)
        # self.fc_c = nn.Linear(in_features=self.seq_len * self.n_feat + self.n_layer * self.n_emb,
        #                       out_features=self.n_layer * self.n_emb)

    def forward(self, x):
        batchsize = x.shape[0]
        x = x.float()
        pharma = x[:, :self.input_len, 1:10]
        x = x[:, :, 10:]

        enc_out, enc_hn = self.encoder(x)
        enc_output = self.fc_enc(enc_out[:, -1, :])
        # enc_output = F.relu(enc_output)

        attn_output, _ = self.att(pharma, x, x, need_weights=False)
        # print(f'000 attn_output: {attn_output.shape}')
        attn_output = self.fc_att(attn_output.reshape(batchsize, -1))
        # attn_output = F.relu(attn_output)
        # print(f'111 enc_output: {enc_output.shape}, attn_output: {attn_output.shape}')

        fc_in = torch.concat((attn_output, enc_output), dim=1)
        # fc_in = F.relu(fc_in)
        # print(f'222 fc_in: {fc_in.shape}')
        dec_in = self.fc2dec(fc_in)
        # print(f'333 dec_in: {dec_in.shape}')

        x_ = self.decoder(dec_in)
        # print(f'444 output x: {x_.shape}')

        return x_

    def training_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x)
        # loss = F.l1_loss(x_, x)
        loss = mae_loss(x_, x[:, :, 10:])

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](x_, x[:, :, 10:]), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x)
        # loss = F.l1_loss(x_, x)
        loss = mae_loss(x_, x[:, :, 10:])

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](x_, x[:, :, 10:])
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x)
        # loss = F.l1_loss(x_, x)
        loss = mae_loss(x_, x[:, :, 10:])

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](x_, x[:, :, 10:])
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs




class LSTM_AE_BENCHMARK(BaseAE):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = LSTMEncoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)
        # self.decoder = LSTMDecoder_recurrent(config=config, seq_len=self.seq_len, n_feat=self.n_feat, n_emb=self.n_emb,
        #                            n_layer=self.n_layer, dropout=self.dropout)
        # self.decoder = LSTMDecoder(config=config, seq_len=self.seq_len, n_feat=self.n_feat, n_emb=self.n_emb,
        #                            n_layer=self.n_layer, dropout=self.dropout)
        self.decoder = nn.Sequential(OrderedDict([
            ('decoder_fc1', nn.Linear(in_features=self.n_emb, out_features=2 * self.n_emb)),
            ('decoder_relu1', nn.ReLU()),
            ('decoder_dropout1', nn.Dropout(self.dropout)),
            ('decoder_fc2', nn.Linear(in_features=2 * self.n_emb, out_features=8 * self.n_emb)),
            ('decoder_relu2', nn.ReLU()),
            ('decoder_dropout2', nn.Dropout(self.dropout)),
            ('decoder_output', nn.Linear(in_features=8 * self.n_emb, out_features=self.n_feat * self.seq_len))
        ]))


    def forward(self, x):
        batchsize = x.shape[0]

        # RECURRENT LSTM DECODER
        # _, enc_hn = self.encoder(x)
        # x_ = self.decoder(enc_hn)

        # LSTM DECODER
        # out, enc_hn = self.encoder(x)
        # x_ = self.decoder(out[:, -1, :])

        out, enc_hn = self.encoder(x)
        x_ = self.decoder(out[:, -1, :])

        return x_.reshape(batchsize, self.seq_len, self.n_feat)

    def training_step(self, batch, batch_idx):
        x = batch['data']['num']
        x_ = self.forward(x)
        loss = mae_loss(x_, x)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](x_, x), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']['num']
        x_ = self.forward(x)
        loss = mae_loss(x_, x)

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](x_, x)
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']['num']
        x_ = self.forward(x)
        loss = mae_loss(x_, x)

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](x_, x)
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs



class LSTM_VAE_BENCHMARK(BaseAE):
    def __init__(self, config):
        super().__init__(config)
        self.n_latent = config['n_latent']
        self.kl_weight = config['kl_weight']

        self.encoder = LSTMEncoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)
        # self.decoder = LSTMDecoder_recurrent(config=config, seq_len=self.seq_len, n_feat=self.n_feat, n_emb=self.n_emb,
        #                            n_layer=self.n_layer, dropout=self.dropout)

        self.decoder = nn.Sequential(OrderedDict([
            ('decoder_fc1', nn.Linear(in_features=self.n_latent, out_features=2 * self.n_latent)),
            ('decoder_relu1', nn.ReLU()),
            ('decoder_dropout1', nn.Dropout(self.dropout)),
            ('decoder_fc2', nn.Linear(in_features=2 * self.n_latent, out_features=8 * self.n_latent)),
            ('decoder_relu2', nn.ReLU()),
            ('decoder_dropout2', nn.Dropout(self.dropout)),
            ('decoder_output', nn.Linear(in_features=8 * self.n_latent, out_features=self.n_feat * self.seq_len))
        ]))

        self.fc_mean = nn.Linear(in_features=self.n_emb, out_features=self.n_latent)
        self.fc_logvar = nn.Linear(in_features=self.n_emb, out_features=self.n_latent)

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.relu = nn.ReLU()


    def forward(self, x):
        batchsize = x.shape[0]
        enc_out, enc_hn = self.encoder(x)
        x_emb = enc_out[:, -1, :]
        # print(f"111 X_EMB: {x_emb.shape}")

        z_mu = self.fc_mean(x_emb)
        z_mu = self.relu(self.dropout_layer(z_mu))
        z_logvar = self.fc_logvar(x_emb)
        z_logvar = self.relu(self.dropout_layer(z_logvar))
        z = self.reparametize(z_mu, z_logvar)
        # print(f"222 Z: {z.shape}")

        x_ = self.decoder(z)
        # print(f"111 X_HAT: {x_.shape}")

        return {
            'x_hat': x_.reshape(batchsize, self.seq_len, self.n_feat),
            'z': z,
            'z_mu': z_mu,
            'z_logvar': z_logvar
        }

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(self.device)

        z = mu + noise * std
        return z
    def loss_function(self, x, out):
        l_reconst = mae_loss(out['x_hat'], x)
        l_kl = torch.mean(-0.5 * torch.sum(1 + out['z_logvar'] - out['z_mu'].pow(2) - out['z_logvar'].exp(), dim=1), dim=0)
        # print(f"LOSS RECONSTRUCTION: {l_reconst} --- X: {x.shape}, X_HAT_MET: {out['x_hat_med'].shape}")
        # print(f"LOSS KL DIV: {l_kl} --- Z_MU: {out['z_mu'].shape}, Z_LOGVAR: {out['z_logvar'].shape}")

        return l_reconst + self.kl_weight * l_kl, l_reconst, l_kl


    def training_step(self, batch, batch_idx):
        x = batch['data']['num']
        out = self.forward(x)
        loss, l_reconst, l_kl = self.loss_function(x, out)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_reconst", l_reconst, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_kl", l_kl, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](out['x_hat'], x), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def on_train_epoch_end(self):
        # set annealing KL weight for next epoch
        N = self.current_epoch + 1
        self.kl_weight = min(0.001, 0.001 * (N % 10) / 8)
        print(f"KL WEIGHT: {self.kl_weight}")


    def validation_step(self, batch, batch_idx):
        x = batch['data']['num']
        out = self.forward(x)
        loss, l_reconst, l_kl = self.loss_function(x, out)

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_reconst", l_reconst, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_kl", l_kl, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](out['x_hat'], x)
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']['num']
        out = self.forward(x)
        loss, l_reconst, l_kl = self.loss_function(x, out)

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_reconst", l_reconst, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_kl", l_kl, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](out['x_hat'], x)
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs



class GuidedLSTM_AE_BENCHMARK(BaseAE):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = LSTMEncoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)
        # self.decoder = LSTMDecoder(config=config, seq_len=self.seq_len, n_feat=self.n_feat, n_emb=self.n_emb,
        #                            n_layer=self.n_layer, dropout=self.dropout)
        self.decoder = LSTMDecoder_recurrent(config=config, seq_len=self.seq_len, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)

        self.att = nn.MultiheadAttention(embed_dim=self.n_feat_med, kdim=self.n_feat, vdim=self.n_feat,
                                         num_heads=1, dropout=self.dropout, batch_first=True)
        self.fc_att = nn.Linear(in_features=self.seq_len * self.n_feat_med, out_features= self.seq_len * self.n_feat)
        self.fc_h = nn.Linear(in_features=self.seq_len * self.n_feat + self.n_layer * self.n_emb,
                              out_features=self.n_layer * self.n_emb)
        self.fc_c = nn.Linear(in_features=self.seq_len * self.n_feat + self.n_layer * self.n_emb,
                              out_features=self.n_layer * self.n_emb)

    def forward(self, x):
        x_num = x['num'].float()
        x_med = x['med'].float()
        batchsize = x_num.shape[0]

        _, enc_hn = self.encoder(x_num)

        attn_output, _ = self.att(x_med, x_num, x_num, need_weights=False)
        # print(f'000 attn_output: {attn_output.shape}')
        attn_output = self.fc_att(attn_output.reshape(batchsize, -1))
        attn_output = F.relu(attn_output)
        # print(f'111 hn: {enc_hn[0].shape}, cn: {enc_hn[1].shape}, permute:{enc_hn[0].permute(1,0,2).shape}')
        hn = enc_hn[0].permute(1,0,2).reshape(batchsize, -1)
        cn = enc_hn[1].permute(1,0,2).reshape(batchsize, -1)
        # print(f'222 hn: {hn.shape}, cn:{cn.shape}, attn_output: {attn_output.shape}')
        # print(f'333 FC INPUT: {torch.concat((attn_output.reshape(batchsize, -1), hn), dim=1).shape}')
        hn_ = self.fc_h(torch.concat((attn_output.reshape(batchsize, -1), hn), dim=1))
        cn_ = self.fc_c(torch.concat((attn_output.reshape(batchsize, -1), cn), dim=1))
        hn_ = hn_.reshape(batchsize, self.n_layer, self.n_emb).permute(1, 0, 2).contiguous()
        cn_ = cn_.reshape(batchsize, self.n_layer, self.n_emb).permute(1, 0, 2).contiguous()
        # print(f'444 hn_: {hn_.shape}, cn_:{cn_.shape}')

        x_ = self.decoder((hn_, cn_))

        # print(f'555 output x: {x_.shape}')

        return x_

    # def configure_optimizers(self):
    #     adam = optim.Adam(self.parameters(), lr=self.config['lr'])
    #     lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, mode='min', factor=.5, patience=5)
    #     return [adam], {"scheduler": lr_scheduler, "monitor": "train_loss"}
    #     # return adam
    #
    # def training_epoch_end(self, outputs):
    #     lr_sch = self.lr_schedulers()
    #     lr_sch.step(self.trainer.callback_metrics["train_loss"])

    def training_step(self, batch, batch_idx):
        if batch_idx == 570:
            pass
        x = batch['data']
        x_ = self.forward(x)
        # loss = F.l1_loss(x_, x)
        loss = mae_loss(x_, x['num'])

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](x_,  x['num']), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x)
        # loss = F.l1_loss(x_, x)
        loss = mae_loss(x_, x['num'])

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](x_,  x['num'])
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x)
        # loss = F.l1_loss(x_, x)
        loss = mae_loss(x_, x['num'])

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](x_,  x['num'])
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs




class GuidedLSTM_VAE_BENCHMARK(BaseAE):
    def __init__(self, config):
        super().__init__(config)
        self.n_latent = config['n_latent']
        self.kl_weight = config['kl_weight']
        self.smooth_weight = config['smooth_weight']

        self.encoder = LSTMEncoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)
        # self.decoder = LSTMDecoder(config=config, seq_len=self.seq_len, n_feat=self.n_feat, n_emb=self.n_emb,
        #                            n_layer=self.n_layer, dropout=self.dropout)
        # self.decoder = LSTMDecoder_recurrent(config=config, seq_len=self.seq_len, n_feat=self.n_feat, n_emb=self.n_emb,
        #                            n_layer=self.n_layer, dropout=self.dropout)
        self.decoder = nn.Sequential(OrderedDict([
            ('decoder_fc1', nn.Linear(in_features=self.n_latent, out_features=2*self.n_latent)),
            ('decoder_relu1', nn.ReLU()),
            ('decoder_dropout1', nn.Dropout(self.dropout)),
            ('decoder_fc2', nn.Linear(in_features=2*self.n_latent, out_features=8*self.n_latent)),
            ('decoder_relu2', nn.ReLU()),
            ('decoder_dropout2', nn.Dropout(self.dropout)),
            ('decoder_output', nn.Linear(in_features=8*self.n_latent, out_features=self.n_feat*self.seq_len))
        ]))

        self.fc_mean = nn.Linear(in_features=self.n_emb, out_features=self.n_latent)
        self.fc_logvar = nn.Linear(in_features=self.n_emb, out_features=self.n_latent)

        self.att = nn.MultiheadAttention(embed_dim=self.n_feat_med, kdim=self.n_feat, vdim=self.n_feat,
                                         num_heads=1, dropout=self.dropout, batch_first=True)
        self.fc_att = nn.Linear(in_features=self.seq_len * self.n_feat_med, out_features=self.n_latent)

        self.fc_mean_med = nn.Linear(in_features=2 * self.n_latent, out_features=self.n_latent)
        self.fc_logvar_med = nn.Linear(in_features=2 * self.n_latent, out_features=self.n_latent)

        # self.fc_h = nn.Linear(in_features=self.seq_len * self.n_feat + self.n_layer * self.n_emb,
        #                       out_features=self.n_layer * self.n_emb)
        # self.fc_c = nn.Linear(in_features=self.seq_len * self.n_feat + self.n_layer * self.n_emb,
        #                       out_features=self.n_layer * self.n_emb)

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.relu = nn.ReLU()


    def forward(self, x):
        x_num = x['num'].float()
        x_med = x['med'].float()
        batchsize = x_num.shape[0]

        # LSTM encoder
        enc_out, enc_hn = self.encoder(x_num)
        x_emb = enc_out[:, -1, :]
        # print(f"111 X_EMB: {x_emb.shape}")
        z_mu = self.fc_mean(x_emb)
        z_mu = self.relu(self.dropout_layer(z_mu))
        z_logvar = self.fc_logvar(x_emb)
        z_logvar = self.relu(self.dropout_layer(z_logvar))
        z = self.reparametize(z_mu, z_logvar)
        # print(f"222 Z: {z.shape}, Z_MU: {z_mu.shape}, Z_LOGVAR: {z_logvar.shape}")

        # Medication attention
        attn_output, _ = self.att(x_med, x_num, x_num, need_weights=False)
        # print(f'000 attn_output: {attn_output.shape}')
        attn_output = self.fc_att(attn_output.reshape(batchsize, -1))
        attn_output = F.relu(attn_output)
        # print(f"333 ATTN_OUTPUT: {attn_output.shape}")

        # Medication-aware reconstruction
        x_emb_med = torch.concat((z, attn_output.reshape(batchsize, -1)), dim=1)
        # print(f"444 X_EMB_MED: {x_emb_med.shape}")
        z_mu_med = self.fc_mean_med(x_emb_med)
        z_mu_med = self.relu(self.dropout_layer(z_mu_med))
        z_logvar_med = self.fc_logvar_med(x_emb_med)
        z_logvar_med = self.relu(self.dropout_layer(z_logvar_med))
        z_med = self.reparametize(z_mu_med, z_logvar_med)
        # print(f"555 Z_MED: {z_med.shape}, Z_MU_MED: {z_mu_med.shape}, Z_LOGVAR_MED: {z_logvar_med.shape}")


        x_hat = self.decoder(z)
        x_hat_med = self.decoder(z_med)
        # print(f'666 X_HAT: {x_hat.shape}')

        return {
            'x_hat': x_hat.reshape(batchsize, self.seq_len, self.n_feat),
            'x_hat_med': x_hat_med.reshape(batchsize, self.seq_len, self.n_feat),
            'z_diff': z_med - z,
            'z_mu': z_mu,
            'z_logvar': z_logvar,
            'z_mu_med': z_mu_med,
            'z_logvar_med': z_logvar_med,
        }

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(self.device)

        z = mu + noise * std
        return z

    def loss_function(self, x, out):
        l_reconst = mae_loss(out['x_hat_med'], x)
        l_kl = torch.mean(-0.5 * torch.sum(1 + out['z_logvar'] - out['z_mu'].pow(2) - out['z_logvar'].exp(), dim=1), dim=0)
        l_smooth = torch.mean(out['z_diff'] ** 2)
        # print(f"LOSS RECONSTRUCTION: {l_reconst} --- X: {x.shape}, X_HAT_MET: {out['x_hat_med'].shape}")
        # print(f"LOSS KL DIV: {l_kl} --- Z_MU: {out['z_mu'].shape}, Z_LOGVAR: {out['z_logvar'].shape}")
        # print(f"LOSS SMOOTH: {l_smooth} --- Z_DIFF: {out['z_diff'].shape}")

        return l_reconst + self.kl_weight * l_kl + self.smooth_weight * l_smooth

    def training_step(self, batch, batch_idx):
        x = batch['data']
        out = self.forward(x)
        # LOSSES
        loss = self.loss_function(x['num'], out)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](out['x_hat_med'],  x['num']), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        out = self.forward(x)

        # LOSSES
        loss = self.loss_function(x['num'], out)

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](out['x_hat_med'],  x['num'])
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']
        out = self.forward(x)

        # LOSSES
        loss = self.loss_function(x['num'], out)

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](out['x_hat_med'],  x['num'])
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs






# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)
#
#
# class Transformer_AE(LightningModule):
#     ### TODOOO: TRAINING LOSS STUCK VERY HIGH
#     def __init__(self, config):
#         super().__init__()
#
#         self.config = config
#         self.seq_len = config['seq_len']
#         self.n_feat = config['n_feat']
#         self.n_emb = config['n_emb']
#         self.n_layer = config['n_layer']
#         self.n_head = config['n_head']
#         self.dmodel = config['dmodel']
#         self.dropout = config['dropout']
#
#         self.encoder = LSTMEncoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
#                                    n_layer=self.n_layer, dropout=self.dropout)
#         self.pos_encoder = PositionalEncoding(self.n_emb)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_emb, nhead=self.n_head, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.n_layer)
#         # self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.n_emb, nhead=self.n_head, batch_first=True)
#         # self.transformer_decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=self.n_layer)
#         # self.decoder = LSTMDecoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
#         #                            n_layer=self.n_layer, dropout=self.dropout)
#         self.fc1 = nn.Linear(in_features=self.n_emb, out_features=512)
#         self.fc2 = nn.Linear(in_features=512, out_features=self.n_feat*self.seq_len)
#
#         self.METRICS = {
#             'mse': mse_loss,
#         }
#
#
#     def forward(self, x):
#         x = x[:,:,2:].float()
#         batchsize = x.shape[0]
#
#         x_enc, _ = self.encoder(x)
#         # print(f'LSTM Encoder: out - {x_enc.shape}')
#         x_enc_pos = self.pos_encoder(x_enc)
#         # print(f'POS Encoder: out - {x_enc.shape}')
#         x_enc_t = self.transformer_encoder(x_enc_pos)
#         # print(f'TRANSFORMER Encoder: out - {x_enc.shape}')
#         x_out = self.fc1(x_enc_t[:, -1, :])
#         x_out = F.relu(x_out)
#         x_out = self.fc2(x_out)
#         x_out = F.sigmoid(x_out)  # check if it worsens/improves the performance
#         x_out = x_out.reshape(batchsize, self.seq_len, self.n_feat)
#         # print(f'OUTPUT: out - {x_out.shape}')
#
#         # x_dec = self.decoder(x_enc_t)
#         # print(f'LSTM Decoder: out - {x_dec.shape}')
#
#         # x_dec = self.transformer_decoder(x_enc, x_enc_t)
#         # print(f'TRANSFORMER Decoder: out - {x_dec.shape}')
#
#
#         return x_out
#
#     def configure_optimizers(self):
#         adam = optim.Adam(self.parameters(), lr=self.config['lr'])
#         lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, mode='min', factor=.5, patience=5)
#         return [adam], {"scheduler": lr_scheduler, "monitor": "train_loss"}
#         # return adam
#
#     def training_epoch_end(self, outputs):
#         lr_sch = self.lr_schedulers()
#         lr_sch.step(self.trainer.callback_metrics["train_loss"])
#
#     def training_step(self, batch, batch_idx):
#         x = batch['data']
#         x_ = self.forward(x)
#         # loss = F.l1_loss(x_, x)
#         loss = mae_loss(x_, x)
#
#         self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         for metric in self.METRICS:
#             self.log("train_" + metric, self.METRICS[metric](x_, x), on_step=False, on_epoch=True, prog_bar=True,
#                      logger=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         x = batch['data']
#         x_ = self.forward(x)
#         # loss = F.l1_loss(x_, x)
#         loss = mae_loss(x_, x)
#
#         outputs = {'val_loss': loss}
#         self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         for metric in self.METRICS:
#             outputs["val_" + metric] = self.METRICS[metric](x_, x)
#             self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
#                      logger=True)
#         return outputs
#
#     def test_step(self, batch, batch_idx):
#         x = batch['data']
#         x_ = self.forward(x)
#         # loss = F.l1_loss(x_, x)
#         loss = mae_loss(x_, x)
#
#         outputs = {'test_loss': loss}
#         self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         for metric in self.METRICS:
#             outputs["test_" + metric] = self.METRICS[metric](x_, x)
#             self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
#                      logger=True)
#         return outputs


