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

from utils.ClassAE import LSTM_AE_ALLMED, GuidedLSTM_AE_ALLMED


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


class BasePRED(LightningModule):
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



class LSTM_PRED(BasePRED):
    def __init__(self, config):
        super().__init__(config)

        # self.config = config
        # self.seq_len = config['seq_len']
        # self.n_feat = config['n_feat']
        # self.n_emb = config['n_emb']
        # self.n_layer = config['n_layer']
        # self.dropout = config['dropout']

        self.lstm = nn.LSTM(
            input_size=self.n_feat,
            hidden_size=self.n_emb,
            num_layers=self.n_layer,
            dropout=self.dropout,
            batch_first=True
        )

        self.fc = nn.Linear(in_features=self.n_emb, out_features=self.n_feat)

        # self.METRICS = {
        #     'mse': mse_loss,
        # }

    def forward(self, x):
        x = x.float()
        len_pred = x.shape[1] - 1

        # set h0 c0 to zero???
        out, _ = self.lstm(x)
        # ADD AN ACTIVATION LAYER???
        preds = self.fc(out[:, -1, :])
        preds = preds[:, None, :]
        # print(f"preds: {preds.shape}")

        for i in range(1, len_pred):
            xin = torch.cat((x[:, i:, :], preds), dim=1)
            # print(f"LSTM input: {xin.shape}")
            out, _ = self.lstm(xin)
            pred = self.fc(out[:, -1, :])
            pred = pred[:, None, :]
            preds = torch.concat((preds, pred), dim=1)
            # print(f"preds: {preds.shape}")

        return preds

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
        x_ = self.forward(x[:, :91, 2:])
        # print(f"loss x: {x[:, 91:, :].shape}, x_: {x_.shape}")
        loss = mae_loss(x_, x[:, 91:, :])

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](x_, x[:, 91:, :]), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x[:, :91, 2:])
        # print(f"loss x: {x[:, 91:, :].shape}, x_: {x_.shape}")
        loss = mae_loss(x_, x[:, 91:, :])

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](x_, x[:, 91:, :])
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x[:, :91, 2:])
        loss = mae_loss(x_, x[:, 91:, :])

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](x_, x[:, 91:, :])
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs


class GuidedLSTM_PRED(BasePRED):
    def __init__(self, config):
        super().__init__(config)

        # self.config = config
        # self.seq_len = config['seq_len']
        # self.n_feat = config['n_feat']
        # self.n_emb = config['n_emb']
        # self.n_layer = config['n_layer']
        # self.dropout = config['dropout']
        # self.input_len = int(self.seq_len//2 + 1)
        # self.pred_len = int(self.seq_len//2)

        self.lstm = nn.LSTM(
            input_size=self.n_feat,
            hidden_size=self.n_emb,
            num_layers=self.n_layer,
            dropout=self.dropout,
            batch_first=True
        )
        self.att = nn.MultiheadAttention(embed_dim=1, kdim=self.n_feat, vdim=self.n_feat,
                                         num_heads=1, dropout=self.dropout, batch_first=True)
        self.fc_lstm = nn.Linear(in_features=self.n_emb, out_features=self.n_feat)
        self.fc_att = nn.Linear(in_features=1 * self.input_len, out_features=self.n_feat * self.pred_len)
        # self.fc_med = nn.Linear(in_features=1, out_features=int(self.seq_len//2))
        # self.fc_emb = nn.Linear(in_features=2 * self.n_feat * self.input_len, out_features=self.n_emb )
        self.fc_out = nn.Linear(in_features=2 * self.n_feat * self.pred_len, out_features=self.n_feat * self.pred_len)

        # self.METRICS = {
        #     'mse': mse_loss,
        # }

    def forward(self, x):
        batchsize = x.shape[0]
        x = x.float()
        pharma_mask = x[:, :, 0]
        dosage = x[:, :, 1]
        dosage = dosage[:, :, None]
        x = x[:, :, 2:]
        len_pred = x.shape[1] - 1


        # set h0 c0 to zero???
        out, _ = self.lstm(x)
        # ADD AN ACTIVATION LAYER???
        preds = self.fc_lstm(out[:, -1, :])
        preds = preds[:, None, :]
        # print(f"preds: {preds.shape}")

        for i in range(1, len_pred):
            xin = torch.cat((x[:, i:, :], preds), dim=1)
            # print(f"LSTM input: {xin.shape}")
            out, _ = self.lstm(xin)
            pred = self.fc_lstm(out[:, -1, :])
            pred = pred[:, None, :]
            preds = torch.concat((preds, pred), dim=1)
            # print(f"preds: {preds.shape}")
        # print(self.att.k_proj_weight.shape, self.att.q_proj_weight.shape)
        # print(f'Dosage: {dosage.shape}, vitals: {x.shape}')
        attn_output, _ = self.att(dosage, x, x, need_weights=False)
        # print(f'ATTN output shape: {attn_output.shape}')
        # print(f'ATTN output: {attn_output}')

        attn_output = self.fc_att(attn_output.reshape(batchsize, -1))
        # print(f'FC_OUT: att - {attn_output.shape}, lstm- {preds.shape}')
        fc_in = torch.concat((attn_output, preds.reshape(batchsize, -1)), dim=1)
        # print(f'FC_OUT: fc_in - {fc_in.shape}')
        output = self.fc_out(fc_in)
        # print(f'FC_OUT: fc_out - {output.shape}')
        output = output.reshape(batchsize, self.pred_len, self.n_feat)
        # print(f'FC_OUT: fc_out_reshaped - {output.shape}')

        return output

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
        x_ = self.forward(x[:, :91, :])
        # print(f"loss x: {x[:, 91:, :].shape}, x_: {x_.shape}")
        loss = mae_loss(x_, x[:, 91:, :])

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](x_, x[:, 91:, :]), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x[:, :91, :])
        # print(f"loss x: {x[:, 91:, :].shape}, x_: {x_.shape}")
        loss = mae_loss(x_, x[:, 91:, :])

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](x_, x[:, 91:, :])
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x[:, :91, :])
        loss = mae_loss(x_, x[:, 91:, :])

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](x_, x[:, 91:, :])
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs


class LSTM_PRED_ALLMED(BasePRED):
    def __init__(self, config):
        super().__init__(config)

        self.include_med = config['include_med']
        if self.include_med:
            self.input_dim = self.n_feat + 9
        else:
            self.input_dim = self.n_feat

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.n_emb,
            num_layers=self.n_layer,
            dropout=self.dropout,
            batch_first=True
        ).to(self.config['device'])

        self.fc = nn.Linear(in_features=self.n_emb, out_features=self.n_feat).to(self.config['device'])


    def forward(self, x):
        x = x.float()
        batchsize = x.shape[0]
        ### TODOOOOO: FIX BUG INCLUDE_MED
        if self.include_med:
            x = x[:, :, 1:]
            x_med = x
        else:
            x = x[:, :, 10:]

        # set h0 c0 to zero???
        h0 = torch.zeros(self.n_layer, batchsize, self.n_emb, device=self.config['device'])
        c0 = torch.zeros(self.n_layer, batchsize, self.n_emb, device=self.config['device'])
        out, _ = self.lstm(x, (h0, c0))
        # ADD AN ACTIVATION LAYER???
        preds = self.fc(out[:, -1, :])
        if self.include_med:
            preds = torch.concat((torch.zeros(batchsize, 9, device=self.config['device']), preds), dim=1)
        # preds = F.sigmoid(preds)
        preds = preds[:, None, :]
        # print(f"preds: {preds.shape}")

        for i in range(1, self.pred_len):
            xin = torch.cat((x[:, i:, :], preds), dim=1)
            # print(f"LSTM input: {xin.shape}")
            out, _ = self.lstm(xin)
            pred = self.fc(out[:, -1, :])
            # print(f'PRED_BEFORE: {pred.shape} - {pred}')
            if self.include_med:
                pred = torch.concat((torch.zeros(batchsize, 9, device=self.config['device']), pred), dim=1)
            # print(f'PRED_AFTER: {pred.shape} - {pred}')
            # pred = F.sigmoid(pred)
            pred = pred[:, None, :]
            preds = torch.concat((preds, pred), dim=1)
            # print(f"preds: {preds.shape}")

        if self.include_med:
            preds = preds[:,:,9:]
        # print(f"FINAL preds: {preds.shape}")

        return preds

    def training_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x[:, :91, :])
        # print(f"loss x: {x[:, 91:, :].shape}, x_: {x_.shape}")
        loss = mae_loss(x_, x[:, 91:, 10:])

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](x_, x[:, 91:, 10:]), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x[:, :91, :])
        # print(f"loss x: {x[:, 91:, :].shape}, x_: {x_.shape}")
        loss = mae_loss(x_, x[:, 91:, 10:])

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](x_, x[:, 91:, 10:])
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']
        x_ = self.forward(x[:, :91, :])
        loss = mae_loss(x_, x[:, 91:, 10:])

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](x_, x[:, 91:, 10:])
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

class GuidedLSTM_PRED_ALLMED(BasePRED):
    def __init__(self, config):
        super().__init__(config)


        self.lstm = nn.LSTM(
            input_size=self.n_feat,
            hidden_size=self.n_emb,
            num_layers=self.n_layer,
            dropout=self.dropout,
            batch_first=True
        ).to(self.config['device'])
        self.att = nn.MultiheadAttention(embed_dim=9, kdim=self.n_feat, vdim=self.n_feat,
                                         num_heads=1, dropout=self.dropout, batch_first=True).to(self.config['device'])
        self.fc_lstm = nn.Linear(in_features=self.n_emb, out_features=self.n_feat).to(self.config['device'])
        self.fc_att = nn.Linear(in_features=9 * self.input_len, out_features=self.n_feat * self.pred_len).to(self.config['device'])
        self.fc_out = nn.Linear(in_features=2 * self.n_feat * self.pred_len, out_features=self.n_feat * self.pred_len).to(self.config['device'])


    def forward(self, x):
        batchsize = x.shape[0]
        x = x.float()
        pharma_mask = x[:, :, 0]
        pharma = x[:, :, 1:10]
        x = x[:, :, 10:]


        # set h0 c0 to zero???
        h0 = torch.zeros(self.n_layer, batchsize, self.n_emb, device=self.config['device'])
        c0 = torch.zeros(self.n_layer, batchsize, self.n_emb, device=self.config['device'])
        # print(self.config['device'])
        # print(x.device)
        # print(h0.device)
        out, _ = self.lstm(x, (h0, c0))
        # ADD AN ACTIVATION LAYER???
        preds = self.fc_lstm(out[:, -1, :])
        # preds = F.sigmoid(preds)
        preds = preds[:, None, :]
        # print(f"preds: {preds.shape}")

        for i in range(1, self.pred_len):
            xin = torch.cat((x[:, i:, :], preds), dim=1)
            # print(f"LSTM input: {xin.shape}")
            out, _ = self.lstm(xin)
            pred = self.fc_lstm(out[:, -1, :])
            pred = pred[:, None, :]
            preds = torch.concat((preds, pred), dim=1)
            # print(f"preds: {preds.shape}")
        # print(self.att.k_proj_weight.shape, self.att.q_proj_weight.shape)
        # print(f'Dosage: {pharma.shape}, vitals: {x.shape}')
        attn_output, _ = self.att(pharma, x, x, need_weights=False)
        # print(f'ATTN output shape: {attn_output.shape}')


        attn_output = self.fc_att(attn_output.reshape(batchsize, -1))
        # print(f'FC_OUT: att - {attn_output.shape}, lstm- {preds.shape}')
        fc_in = torch.concat((attn_output, preds.reshape(batchsize, -1)), dim=1)
        # print(f'FC_OUT: fc_in - {fc_in.shape}')
        output = self.fc_out(fc_in)
        # print(f'FC_OUT: fc_out - {output.shape}')
        output = output.reshape(batchsize, self.pred_len, self.n_feat)
        # print(f'FC_OUT: fc_out_reshaped - {output.shape}')

        return output

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
        x = batch['data'].to(self.config['device'])
        x_ = self.forward(x[:, :91, :])
        # print(f"loss x: {x[:, 91:, :].shape}, x_: {x_.shape}")
        loss = mae_loss(x_, x[:, 91:, 10:])

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](x_, x[:, 91:, 10:]), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data'].to(self.config['device'])
        x_ = self.forward(x[:, :91, :])
        # print(f"loss x: {x[:, 91:, :].shape}, x_: {x_.shape}")
        loss = mae_loss(x_, x[:, 91:, 10:])

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](x_, x[:, 91:, 10:])
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data'].to(self.config['device'])
        x_ = self.forward(x[:, :91, :])
        loss = mae_loss(x_, x[:, 91:, 10:])

        outputs = {
            'test_loss': loss,
            'model_output': x_,
        }
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](x_, x[:, 91:, 10:])
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs



class GuidedLSTM_V2_PRED_ALLMED(BasePRED):
    def __init__(self, config):
        super().__init__(config)


        self.lstm = nn.LSTM(
            input_size=self.n_feat,
            hidden_size=self.n_emb,
            num_layers=self.n_layer,
            dropout=self.dropout,
            batch_first=True
        ).to(self.config['device'])
        self.att1 = nn.MultiheadAttention(embed_dim=9, kdim=self.n_feat, vdim=self.n_feat,
                                         num_heads=1, dropout=self.dropout, batch_first=True).to(self.config['device'])
        self.att2 = nn.MultiheadAttention(embed_dim=9, kdim=self.n_emb, vdim=self.n_emb,
                                          num_heads=1, dropout=self.dropout, batch_first=True).to(self.config['device'])

        self.fc_lstm = nn.Linear(in_features=self.n_emb, out_features=self.n_feat).to(self.config['device'])
        self.fc_att = nn.Linear(in_features=9 * self.input_len, out_features=self.n_feat * self.pred_len).to(self.config['device'])
        self.fc_out = nn.Linear(in_features=2 * self.n_feat * self.pred_len, out_features=self.n_feat * self.pred_len).to(self.config['device'])


    def forward(self, x):
        batchsize = x.shape[0]
        x = x.float()
        pharma_mask = x[:, :, 0]
        pharma = x[:, :, 1:10]
        x = x[:, :, 10:]


        # set h0 c0 to zero???
        h0 = torch.zeros(self.n_layer, batchsize, self.n_emb, device=self.config['device'])
        c0 = torch.zeros(self.n_layer, batchsize, self.n_emb, device=self.config['device'])
        # print(self.config['device'])
        # print(x.device)
        # print(h0.device)

        out, _ = self.lstm(x, (h0, c0))
        lstm_outputs = out[:, -1, :]
        lstm_outputs = lstm_outputs[:, None, :]
        # ADD AN ACTIVATION LAYER???
        preds = self.fc_lstm(out[:, -1, :])
        # preds = F.sigmoid(preds)
        preds = preds[:, None, :]
        # print(f"preds: {preds.shape}")

        for i in range(1, self.pred_len):
            xin = torch.cat((x[:, i:, :], preds), dim=1)
            # print(f"LSTM input: {xin.shape}")
            out, _ = self.lstm(xin)
            out_ = out[:, -1, :]
            out_ = out_[:, None, :]
            # print('LSTM_OUTPUTS: ', lstm_outputs.shape)
            # print('OUT_: ', out_.shape)
            lstm_outputs = torch.concat((lstm_outputs, out_), dim=1)
            # print('LSTM OUTPUT: ', out.shape)
            pred = self.fc_lstm(out[:, -1, :])
            pred = pred[:, None, :]
            preds = torch.concat((preds, pred), dim=1)
            # print(f"preds: {preds.shape}")

        # print('LSTM_OUTPUTS: ', lstm_outputs.shape)
        # print(self.att.k_proj_weight.shape, self.att.q_proj_weight.shape)
        # print(f'Dosage: {pharma.shape}, vitals: {x.shape}')
        attn1_output, _ = self.att1(pharma, x, x, need_weights=False)
        # print(f'ATTN output shape: {attn1_output.shape}')
        attn2_output, _ = self.att2(attn1_output, lstm_outputs, lstm_outputs, need_weights=False)
        # print('ATTN2: ', attn2_output.shape)

        attn_output = self.fc_att(attn2_output.reshape(batchsize, -1))
        # print(f'FC_OUT: att - {attn_output.shape}, lstm- {preds.shape}')
        fc_in = torch.concat((attn_output, preds.reshape(batchsize, -1)), dim=1)
        # print(f'FC_OUT: fc_in - {fc_in.shape}')
        output = self.fc_out(fc_in)
        # print(f'FC_OUT: fc_out - {output.shape}')
        output = output.reshape(batchsize, self.pred_len, self.n_feat)
        # print(f'FC_OUT: fc_out_reshaped - {output.shape}')

        return output

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
        x = batch['data'].to(self.config['device'])
        x_ = self.forward(x[:, :91, :])
        # print(f"loss x: {x[:, 91:, :].shape}, x_: {x_.shape}")
        loss = mae_loss(x_, x[:, 91:, 10:])

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](x_, x[:, 91:, 10:]), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data'].to(self.config['device'])
        x_ = self.forward(x[:, :91, :])
        # print(f"loss x: {x[:, 91:, :].shape}, x_: {x_.shape}")
        loss = mae_loss(x_, x[:, 91:, 10:])

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](x_, x[:, 91:, 10:])
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data'].to(self.config['device'])
        x_ = self.forward(x[:, :91, :])
        loss = mae_loss(x_, x[:, 91:, 10:])

        outputs = {
            'test_loss': loss,
            'model_output': x_,
        }
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](x_, x[:, 91:, 10:])
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs



class PRETRAINED_PRED_ALLMED(BasePRED):
    def __init__(self, config, model):
        super().__init__(config)
        self.dropout_pred = config['dropout_pred']
        self.lr_pred = config['lr_pred']

        self.pretrained_model = model
        self.pretrained_model.to(self.config['device'])
        self.pretrained_model.eval()

        if isinstance(self.pretrained_model, GuidedLSTM_AE_ALLMED):
            self.encoder = self.pretrained_model.encoder
            # self.decoder = self.pretrained_model.decoder
            self.att = self.pretrained_model.att
            self.fc_enc = self.pretrained_model.fc_enc
            # self.fc_att = self.pretrained_model.fc_att
            # self.fc2dec = self.pretrained_model.fc2dec
        elif isinstance(self.pretrained_model, LSTM_AE_ALLMED):
            self.encoder = self.pretrained_model.encoder
            # self.decoder = self.pretrained_model.decoder
        else:
            raise ValueError(f"{type(self.pretrained_model)} IS NOT SUPPORTED")


        self.fc1 = nn.Linear(in_features=self.n_emb, out_features=int(self.n_emb/2)).to(self.config['device'])
        self.fc2 = nn.Linear(in_features=int(self.n_emb/2), out_features=10).to(self.config['device'])
        self.dropout = nn.Dropout(self.dropout_pred)
        self.fc_att = nn.Linear(in_features=9 * self.input_len, out_features=self.n_feat * self.pred_len)
        self.fc_out = nn.Linear(in_features=2 * self.n_feat * self.pred_len,
                                out_features=self.n_feat * self.pred_len).to(self.config['device'])


    def forward(self, x):
        batchsize = x.shape[0]
        x = x.float()
        pharma_mask = x[:, :, 0]
        pharma = x[:, :, 1:10]
        x = x[:, :, 10:]

        if isinstance(self.pretrained_model, LSTM_AE_ALLMED):
            out, _ = self.encoder(x)
            preds = self.fc1(out[:, -1, :])
            preds = torch.relu(preds)
            preds = self.dropout(preds)
            # if self.include_med:
            #     preds = torch.concat((torch.zeros(batchsize, 9, device=self.config['device']), preds), dim=1)
            preds = self.fc2(preds)
            preds = preds[:, None, :]

            # print(f"preds: {preds.shape}")

            for i in range(1, self.pred_len):
                xin = torch.cat((x[:, i:, :], preds), dim=1)
                # print(f"LSTM input: {xin.shape}")
                out, _ = self.encoder(xin)
                pred = self.fc1(out[:, -1, :])
                pred = torch.relu(pred)
                pred = self.dropout(pred)
                pred = self.fc2(pred)
                # print(f'PRED_BEFORE: {pred.shape} - {pred}')
                # if self.include_med:
                #     pred = torch.concat((torch.zeros(batchsize, 9, device=self.config['device']), pred), dim=1)
                # print(f'PRED_AFTER: {pred.shape} - {pred}')
                # pred = F.sigmoid(pred)
                pred = pred[:, None, :]
                preds = torch.concat((preds, pred), dim=1)
                # print(f"preds: {preds.shape}")
            output = preds

        elif isinstance(self.pretrained_model, GuidedLSTM_AE_ALLMED):
            lstm_output, _ = self.encoder(x)
            lstm_output = self.fc_enc(lstm_output[:, -1, :])
            # attn_output, _ = self.att(pharma, x, x, need_weights=False)
            # attn_output = self.fc_att(attn_output.reshape(batchsize, -1))
            #
            # out = torch.concat((attn_output, lstm_output), dim=1)
            # out = self.fc2dec(out)

            # preds = self.fc1(out)
            preds = self.fc1(lstm_output)
            preds = torch.relu(preds)
            preds = self.dropout(preds)
            preds = self.fc2(preds)
            preds = preds[:, None, :]

            for i in range(1, self.pred_len):
                xin = torch.cat((x[:, i:, :], preds), dim=1)
                # pharma_in = torch.cat((pharma[:, i:, :], torch.zeros(batchsize, i, 9, device=self.config['device'])), dim=1)
                # print(f"LSTM input: {xin.shape}")
                # print(f"PHARMA_IN: {pharma_in.shape}")
                lstm_output, _ = self.encoder(xin)
                lstm_output = self.fc_enc(lstm_output[:, -1, :])
                # attn_output, _ = self.att(pharma_in, xin, xin, need_weights=False)
                # attn_output = self.fc_att(attn_output.reshape(batchsize, -1))

                # out = torch.concat((attn_output, lstm_output), dim=1)
                # out = self.fc2dec(out)

                # pred = self.fc1(out)
                pred = self.fc1(lstm_output)
                pred = torch.relu(pred)
                pred = self.dropout(pred)
                pred = self.fc2(pred)
                pred = pred[:, None, :]
                preds = torch.concat((preds, pred), dim=1)

            attn_output, _ = self.att(pharma, x, x, need_weights=False)
            attn_output = self.fc_att(attn_output.reshape(batchsize, -1))
            fc_in = torch.concat((attn_output, preds.reshape(batchsize, -1)), dim=1)
            fc_in = torch.relu(fc_in)
            fc_in = self.dropout(fc_in)
            # print(f'FC_OUT: fc_in - {fc_in.shape}')
            output = self.fc_out(fc_in)
            # print(f'FC_OUT: fc_out - {output.shape}')
            output = output.reshape(batchsize, self.pred_len, self.n_feat)


        return output


    def training_step(self, batch, batch_idx):
        x = batch['data'].to(self.config['device'])
        x_ = self.forward(x[:, :91, :])
        # print(f"loss x: {x[:, 91:, :].shape}, x_: {x_.shape}")
        # print(x_[-1, -1, :5])
        loss = mae_loss(x_, x[:, 91:, 10:])

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](x_, x[:, 91:, 10:]), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data'].to(self.config['device'])
        x_ = self.forward(x[:, :91, :])
        # print(f"loss x: {x[:, 91:, :].shape}, x_: {x_.shape}")
        loss = mae_loss(x_, x[:, 91:, 10:])

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](x_, x[:, 91:, 10:])
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data'].to(self.config['device'])
        x_ = self.forward(x[:, :91, :])
        loss = mae_loss(x_, x[:, 91:, 10:])

        outputs = {
            'test_loss': loss,
            'model_output': x_,
        }
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](x_, x[:, 91:, 10:])
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def configure_optimizers(self):
        adam = optim.Adam(self.parameters(), lr=self.lr_pred)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, mode='min', factor=.5, patience=5)
        return [adam], {"scheduler": lr_scheduler, "monitor": "train_loss"}






