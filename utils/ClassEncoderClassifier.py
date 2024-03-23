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
    loss = torch.mean(torch.abs(x - x_))
    return loss

def mse_loss(x_, x):
    loss = torch.mean((x - x_)**2)
    return loss


class LSTMEncoder(nn.Module):
    def __init__(self, n_feat=10, n_emb=128, n_layer=2, dropout=0):
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

class PretrainedAE(nn.Module):

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.seq_len = config['seq_len']
        self.n_feat = config['n_feat']
        self.n_emb = config['n_emb']
        self.n_layer = config['n_layer']
        self.dropout = config['dropout']
        self.input_len = int(self.seq_len // 2 + 1)
        self.pred_len = int(self.seq_len // 2)


        self.encoder = model.encoder
        self.decoder = model.decoder



    def forward(self, x):
        x = x.float()
        x = x[:, :, -10:]

        out, _ = self.encoder(x)
        x_emb = out[:, -1, :]
        x_reconstruct = self.decoder(x_emb)

        return (x_reconstruct, x_emb)


class PretrainedGAE(nn.Module):

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.seq_len = config['seq_len']
        self.n_feat = config['n_feat']
        self.n_emb = config['n_emb']
        self.n_layer = config['n_layer']
        self.dropout = config['dropout']
        self.input_len = int(self.seq_len // 2 + 1)
        self.pred_len = int(self.seq_len // 2)


        self.encoder = model.encoder
        self.decoder = model.decoder
        self.att = model.att
        self.fc_enc = model.fc_enc
        self.fc_att = model.fc_att
        self.fc2dec = model.fc2dec


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
        x_emb = self.fc2dec(fc_in)
        # print(f'333 dec_in: {dec_in.shape}')

        x_reconstruct = self.decoder(x_emb)

        return (x_reconstruct, x_emb)


class BaseClassifier(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len = config['seq_len']
        self.n_feat = config['n_feat']
        self.dropout = config['dropout']
        self.input_len = int(self.seq_len // 2 + 1)
        self.pred_len = int(self.seq_len // 2)

        self.METRICS = {
            # 'acc': torchmetrics.Accuracy().cpu(),
            # 'recall': torchmetrics.Recall().cpu(),
            # 'precision': torchmetrics.Precision().cpu(),
            # 'specification': torchmetrics.Specificity().cpu(),
            # 'roc': torchmetrics.AUROC(pos_label=1).cpu(),
            # 'f1': torchmetrics.F1().cpu(),
        }

    def configure_optimizers(self):
        adam = optim.Adam(self.parameters(), lr=self.config['lr'])
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, mode='min', factor=.5, patience=5)
        return [adam], {"scheduler": lr_scheduler, "monitor": "train_loss"}
        # return adam

    def training_epoch_end(self, outputs):
        lr_sch = self.lr_schedulers()
        lr_sch.step(self.trainer.callback_metrics["train_loss"])




class GuidedMortalityClassification_ALLMED(BaseClassifier):

    def __init__(self, config):

        super().__init__(config)

        # self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.loss = nn.BCEWithLogitsLoss()

        self.METRICS = {
            'roc': torchmetrics.AUROC(pos_label=1).to(self.config['device']),
            'acc': torchmetrics.Accuracy().to(device=self.config['device']),
            'recall': torchmetrics.Recall().to(device=self.config['device']),
            'precision': torchmetrics.Precision().to(device=self.config['device']),
            'f1': torchmetrics.F1().to(self.config['device']),
        }

        self.fc_info = nn.Linear(in_features=6, out_features=16)
        self.cnn_x = nn.Conv1d(in_channels=10, out_channels=16, kernel_size=21, stride=10)
        self.cnn_emb = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=16, stride=8)
        self.fc0_out = nn.Linear(in_features=512, out_features=64)
        # self.fc1_out = nn.Linear(in_features=256, out_features=64)
        self.fc2_out = nn.Linear(in_features=64, out_features=1)
        self.dropout = nn.Dropout(p=config['dropout'])


    def forward(self, x, x_rec, x_enc, x_info):
        # x: batchsize x 181 x 10
        # x_rec: batchsize x 181 x 10
        # x_enc: batchsize x 5 x 128
        # x_info: batchsize x 6

        x = x.float()
        mask = x==-1
        x_rec = x_rec.float()
        x_enc = x_enc.float()
        x_info = x_info.float()
        batchsize = x.shape[0]

        x_diff = (x - x_rec)
        x_diff[mask] = 0
        x_diff = x_diff.permute(0, 2, 1)

        # print(f'X_INFO: {x_info.shape}')
        # x_info = self.fc_info(x_info)
        # print(f'X_INFO EMB: {x_info.shape}')
        # print(f'X_DIFF: {x_diff.shape}')
        x_diff = self.cnn_x(x_diff)
        # print(f'X_DIFF EMB: {x_diff.shape}')
        # print(f'X_ENC: {x_enc.shape}')
        x_emb = self.cnn_emb(x_enc)
        # print(f'X_ENC EMB: {x_emb.shape}')
        # x_concat = torch.concat((x_info, x_diff.reshape(batchsize, -1), x_emb.reshape(batchsize, -1)), dim=1)
        x_concat = torch.concat((x_diff.reshape(batchsize, -1), x_emb.reshape(batchsize, -1)), dim=1)
        # print(f'X_CONCAT: {x_concat.shape}')
        x_out = self.fc0_out(x_concat)
        x_out = F.relu(x_out)
        x_out = self.dropout(x_out)
        # x_out = self.fc1_out(x_out)
        # # print(f'X_FC1: {x_out.shape}')
        # x_out = F.relu(x_out)
        # x_out = self.dropout(x_out)
        logit = self.fc2_out(x_out)
        # print(f'X_FC2: {logit.shape}')
        prob = F.sigmoid(logit)

        return logit, prob


    def training_step(self, batch, batch_idx):
        x, x_rec, x_enc = batch['data']
        batchsize = x.shape[0]
        info = batch['info']
        label = batch['label']

        x_info = torch.concat((info['age'][:, None],
                               info['sex'].reshape(batchsize, -1),
                               info['apache'].reshape(batchsize, -1)),
                              dim=1)
        y = label.reshape(batchsize, 1)
        # print(y[y==1].size())
        logit, prob = self.forward(x[:, :, 10:], x_rec, x_enc, x_info)
        loss = self.loss(logit, y.float())

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](prob, y), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_rec, x_enc = batch['data']
        batchsize = x.shape[0]
        info = batch['info']
        label = batch['label']
        x_info = torch.concat((info['age'][:, None],
                               info['sex'].reshape(batchsize, -1),
                               info['apache'].reshape(batchsize, -1)),
                              dim=1)
        y = label.reshape(batchsize, 1)
        # print(y[y == 1].size())
        logit, prob = self.forward(x[:,:,10:], x_rec, x_enc, x_info)
        loss = self.loss(logit, y.float())

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            # print(prob.device)
            # print(y.device)
            # print(self.METRICS[metric].device)
            outputs["val_" + metric] = self.METRICS[metric](prob, y)
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x, x_rec, x_enc = batch['data']
        batchsize = x.shape[0]
        info = batch['info']
        label = batch['label']
        x_info = torch.concat((info['age'][:, None],
                               info['sex'].reshape(batchsize, -1),
                               info['apache'].reshape(batchsize, -1)),
                              dim=1)
        y = label.reshape(batchsize, 1)
        logit, prob = self.forward(x[:, :, 10:], x_rec, x_enc, x_info)
        loss = self.loss(logit, y.float())

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](prob, y)
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs


class GuidedLOS_PRED_ALLMED(BaseClassifier):

    def __init__(self, config):

        super().__init__(config)
        self.n_emb = config['n_emb']

        self.METRICS = {
            'mse': mse_loss,
        }

        self.loss = mae_loss

        self.fc_info = nn.Linear(in_features=6, out_features=16)
        self.cnn_x = nn.Conv1d(in_channels=10, out_channels=16, kernel_size=21, stride=10)
        if self.n_emb == 128:
            self.cnn_emb = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=16, stride=8)
        elif self.n_emb == 256:
            self.cnn_emb = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=32, stride=16)
        else:
            raise ValueError(f"Only support data embedding of size 128 or 256.")
        self.fc0_out = nn.Linear(in_features=512, out_features=256)
        # self.fc0_out = nn.Linear(in_features=528, out_features=256)
        self.fc1_out = nn.Linear(in_features=256, out_features=64)
        self.fc2_out = nn.Linear(in_features=64, out_features=1)
        self.dropout = nn.Dropout(p=config['dropout'])


    def forward(self, x, x_rec, x_enc, x_info):
        # x: batchsize x 181 x 10
        # x_rec: batchsize x 181 x 10
        # x_enc: batchsize x 5 x 128
        # x_info: batchsize x 6

        x = x.float()
        mask = x == -1
        x_rec = x_rec.float()
        x_enc = x_enc.float()
        x_info = x_info.float()
        batchsize = x.shape[0]

        x_diff = (x - x_rec)
        x_diff[mask] = 0
        x_diff = x_diff.permute(0, 2, 1)

        # print(f'X_INFO: {x_info.shape}')
        # x_info = self.fc_info(x_info)
        # print(f'X_INFO EMB: {x_info.shape}')
        # print(f'X_DIFF: {x_diff.shape}')
        x_diff = self.cnn_x(x_diff)
        # print(f'X_DIFF EMB: {x_diff.shape}')
        # print(f'X_ENC: {x_enc.shape}')
        x_emb = self.cnn_emb(x_enc)
        # print(f'X_ENC EMB: {x_emb.shape}')
        # x_concat = torch.concat((x_info, x_diff.reshape(batchsize, -1), x_emb.reshape(batchsize, -1)), dim=1)
        x_concat = torch.concat((x_diff.reshape(batchsize, -1), x_emb.reshape(batchsize, -1)), dim=1)
        # print(f'X_CONCAT: {x_concat.shape}')
        x_out = self.fc0_out(x_concat)
        x_out = F.relu(x_out)
        x_out = self.dropout(x_out)
        x_out = self.fc1_out(x_out)
        # # print(f'X_FC1: {x_out.shape}')
        x_out = F.relu(x_out)
        x_out = self.dropout(x_out)
        pred = self.fc2_out(x_out)
        # print(f'X_FC2: {logit.shape}')

        return pred

    def training_step(self, batch, batch_idx):
        x, x_rec, x_enc = batch['data']
        batchsize = x.shape[0]
        info = batch['info']
        y = batch['label'].reshape(batchsize, 1)
        # print(batch['label'][:5])
        x_info = torch.concat((info['age'][:, None],
                               info['sex'].reshape(batchsize, -1),
                               info['apache'].reshape(batchsize, -1)),
                              dim=1)
        y_ = self.forward(x[:, :, 10:], x_rec, x_enc, x_info)
        loss = self.loss(y_, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](y_, y), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_rec, x_enc = batch['data']
        batchsize = x.shape[0]
        info = batch['info']
        y = batch['label'].reshape(batchsize, 1)
        x_info = torch.concat((info['age'][:, None],
                               info['sex'].reshape(batchsize, -1),
                               info['apache'].reshape(batchsize, -1)),
                              dim=1)
        y_ = self.forward(x[:, :, 10:], x_rec, x_enc, x_info)
        loss = self.loss(y_, y)

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_" + metric] = self.METRICS[metric](y_, y)
            self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x, x_rec, x_enc = batch['data']
        batchsize = x.shape[0]
        info = batch['info']
        y = batch['label'].reshape(batchsize, 1)
        x_info = torch.concat((info['age'][:, None],
                               info['sex'].reshape(batchsize, -1),
                               info['apache'].reshape(batchsize, -1)),
                              dim=1)
        y_ = self.forward(x[:, :, 10:], x_rec, x_enc, x_info)
        loss = self.loss(y_, y)

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](y_, y)
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs





# class LOS_PRED_ALLMED(BaseClassifier):
#
#     def __init__(self, config):
#
#         super().__init__(config)
#
#         self.METRICS = {
#             'mse': mse_loss,
#         }
#
#         self.lstm = nn.LSTM(
#             input_size=self.n_feat,
#             hidden_size=self.n_emb,
#             num_layers=self.n_layer,
#             dropout=self.dropout,
#             batch_first=True
#         )
#
#         self.fc1 = nn.Linear(in_features=self.n_emb, out_features=self.n_emb//2)
#         self.fc2 = nn.Linear(self.n_emb//2, 1)
#
#     def forward(self, x):
#         x = x.float()
#         batchsize = x.shape[0]
#
#         h0 = torch.zeros(self.n_layer, batchsize, self.n_emb).to(self.config['device'])
#         c0 = torch.zeros(self.n_layer, batchsize, self.n_emb).to(self.config['device'])
#         out, _ = self.lstm(x, (h0, c0))
#
#         fc_out = self.fc1(out[:, -1, :])
#         fc_out = F.relu(fc_out)
#         pred = self.fc2(fc_out)
#         # print(f'PRED: {pred.shape}')
#
#         return pred
#
#     def training_step(self, batch, batch_idx):
#         x = batch['data']
#         y = batch['label']['los']
#         y_ = self.forward(x[:, :, 10:])
#         loss = mae_loss(y_, y)
#
#         self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         for metric in self.METRICS:
#             self.log("train_" + metric, self.METRICS[metric](y_, y), on_step=False, on_epoch=True, prog_bar=True,
#                      logger=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         # print(batch['label'])
#         x = batch['data']
#         y = batch['label']['los']
#         y_ = self.forward(x[:, :, 10:])
#         loss = mae_loss(y_, y)
#
#         outputs = {'val_loss': loss}
#         self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         for metric in self.METRICS:
#             outputs["val_" + metric] = self.METRICS[metric](y_, y)
#             self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
#                      logger=True)
#         return outputs
#
#     def test_step(self, batch, batch_idx):
#         x = batch['data']
#         y = batch['label']['los']
#         y_ = self.forward(x[:, :, 10:])
#         loss = mae_loss(y_, y)
#
#         outputs = {'test_loss': loss}
#         self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         for metric in self.METRICS:
#             outputs["test_" + metric] = self.METRICS[metric](y_, y)
#             self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
#                      logger=True)
#         return outputs
#
#
# class GuidedLOS_PRED_ALLMED(BaseClassifier):
#
#     def __init__(self, config):
#
#         super().__init__(config)
#
#         self.METRICS = {
#             'mse': mse_loss,
#         }
#
#         self.lstm = nn.LSTM(
#             input_size=self.n_feat,
#             hidden_size=self.n_emb,
#             num_layers=self.n_layer,
#             dropout=self.dropout,
#             batch_first=True
#         )
#
#         self.att = nn.MultiheadAttention(embed_dim=9, kdim=self.n_feat, vdim=self.n_feat,
#                                          num_heads=1, dropout=self.dropout, batch_first=True)
#
#         self.fc_lstm = nn.Linear(in_features=self.n_emb, out_features=self.n_emb)
#         self.fc_att = nn.Linear(in_features=9 * self.input_len, out_features=self.n_emb)
#
#         self.fc1 = nn.Linear(in_features=self.n_emb * 2, out_features=self.n_emb // 2)
#         self.fc2 = nn.Linear(self.n_emb // 2, 1)
#
#
#     def forward(self, x):
#         x = x.float()
#         batchsize = x.shape[0]
#         pharma = x[:, :91, 1:10]
#         x = x[:, :, 10:]
#
#         h0 = torch.zeros(self.n_layer, batchsize, self.n_emb).to(self.config['device'])
#         c0 = torch.zeros(self.n_layer, batchsize, self.n_emb).to(self.config['device'])
#         out, _ = self.lstm(x, (h0, c0))
#
#         attn_output, _ = self.att(pharma, x, x, need_weights=False)
#         # print(f'ATTN output shape: {attn_output.shape}')
#
#         attn_output = self.fc_att(attn_output.reshape(batchsize, -1))
#         lstm_output = self.fc_lstm(out[:, -1, :].reshape(batchsize, -1))
#         # print(f'FC_OUT: att - {attn_output.shape}, lstm- {lstm_output.shape}')
#         fc_in = torch.concat((attn_output, lstm_output), dim=1)
#
#         fc_out = self.fc1(fc_in)
#         fc_out = F.relu(fc_out)
#         pred = self.fc2(fc_out)
#         # print(f'PRED: {pred.shape}')
#
#         return pred
#
#     def training_step(self, batch, batch_idx):
#         x = batch['data']
#         y = batch['label']['los']
#         y_ = self.forward(x)
#         loss = mae_loss(y_, y)
#
#         self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         for metric in self.METRICS:
#             self.log("train_" + metric, self.METRICS[metric](y_, y), on_step=False, on_epoch=True, prog_bar=True,
#                      logger=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         x = batch['data']
#         y = batch['label']['los']
#         y_ = self.forward(x)
#         loss = mae_loss(y_, y)
#
#         outputs = {'val_loss': loss}
#         self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         for metric in self.METRICS:
#             outputs["val_" + metric] = self.METRICS[metric](y_, y)
#             self.log("val_" + metric, outputs["val_" + metric], on_step=False, on_epoch=True, prog_bar=True,
#                      logger=True)
#         return outputs
#
#     def test_step(self, batch, batch_idx):
#         x = batch['data']
#         y = batch['label']['los']
#         y_ = self.forward(x)
#         loss = mae_loss(y_, y)
#
#         outputs = {'test_loss': loss}
#         self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         for metric in self.METRICS:
#             outputs["test_" + metric] = self.METRICS[metric](y_, y)
#             self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
#                      logger=True)
#         return outputs

