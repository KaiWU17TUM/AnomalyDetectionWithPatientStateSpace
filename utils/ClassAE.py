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


    def forward(self, x):
        batchsize = x.shape[0]

        h0 = torch.zeros(self.n_layer, batchsize, self.n_emb).to('cuda')
        c0 = torch.zeros(self.n_layer, batchsize, self.n_emb).to('cuda')
        # print(f'DECODER LSTM INPUT: {x.shape}')
        x = x[:, None, :].repeat(1, self.seq_len, 1)
        # print(f'DECODER LSTM INPUT: {x.shape}')
        out, _ = self.lstm(x, (h0, c0))
        out = F.relu(out)
        # print(f'DECODER LSTM OUTPUT: {out.shape}')
        x_ = self.out(out)
        # x_ = F.sigmoid(x_)
        # print(f'DECODER FC OUTPUT: {x_.shape}')

        return x_




class BaseAE(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len = config['seq_len']
        self.n_feat = config['n_feat']
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
        self.decoder = LSTMDecoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)

        # self.METRICS = {
        #     'mse': mse_loss,
        # }


    def forward(self, x):
        _, enc_hn = self.encoder(x)
        x_ = self.decoder(enc_hn)

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

        # self.config = config
        # self.seq_len = config['seq_len']
        # self.n_feat = config['n_feat']
        # self.n_emb = config['n_emb']
        # self.n_layer = config['n_layer']
        # self.dropout = config['dropout']
        # self.input_len = int(self.seq_len // 2 + 1)
        # self.pred_len = int(self.seq_len // 2)

        self.encoder = LSTMEncoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)
        self.decoder = LSTMDecoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
                                   n_layer=self.n_layer, dropout=self.dropout)

        # self.METRICS = {
        #     'mse': mse_loss,
        # }


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
        hn_ = self.fc_h(torch.concat((attn_output.reshape(batchsize, -1), hn), dim=1)).contiguous()
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
        self.decoder = LSTMDecoder(config=config, n_feat=self.n_feat, n_emb=self.n_emb,
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


