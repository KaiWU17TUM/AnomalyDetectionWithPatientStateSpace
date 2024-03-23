from pathlib import Path
import time
import glob
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split, KFold

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import ModelSummary

from utils.config_dataset import *
from utils.ClassDataset import CusDataset
from utils.ClassPredictor import LSTM_PRED, GuidedLSTM_PRED, LSTM_PRED_ALLMED, GuidedLSTM_PRED_ALLMED, mae_loss

RANDOMSEED=2024
torch.manual_seed(RANDOMSEED)
np.random.seed(RANDOMSEED)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':
    model_type = 'GuidedLSTM_PRED_ALLMED'
    model_path = 'models/GuidedLSTM_PRED_ALLMED/2layer-128hidden-0.1dropout-64-0.001/version_0'

    include_med = False
    kfolds = 5
    lr = 1e-3
    batchsize = 64
    n_emb = 128
    n_layer = 2
    dropout = 0.1
    device = 'cuda'
    seq_len = 181
    n_feat = 10

    config = {
        'device': device,
        "seq_len": seq_len,
        "n_feat": n_feat,
        "n_emb": n_emb,
        'n_layer': n_layer,
        "lr": lr,
        "dropout": dropout,
        'include_med': include_med,
    }

    archive = pickle.load(open(os.path.join(path_processed, 'training_data_allmed.p'), 'rb'))
    samples = archive['samples_norm']
    sample_dict = archive['sample_dict']
    idx_train = archive['idx_train']
    idx_test = archive['idx_test']
    pid_train = archive['pid_train']
    pid_test = archive['pid_test']
    del archive

    X_test = samples[idx_test]
    dataset_test = CusDataset(X_test)
    loader_test = DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=32)

    model_files = glob.glob(f'{model_path}/*.ckpt')

    preds = {model_file:[] for model_file in model_files}
    losses = {model_file:[] for model_file in model_files}
    for model_file in model_files:
        print('Evaluating: ', model_file)
        print('Save model output to:', model_file.replace('.ckpt', '_test_preds.p'))

        if model_type == 'LSTM_PRED_ALLMED':
            model = LSTM_PRED_ALLMED.load_from_checkpoint(checkpoint_path=model_file, config=config)
        elif model_type == 'GuidedLSTM_PRED_ALLMED':
            model = GuidedLSTM_PRED_ALLMED.load_from_checkpoint(checkpoint_path=model_file, config=config)
        else:
            print(f'{model_type} is not supported!')

        # print(model.lstm.all_weights.type)

        for X in tqdm(loader_test):
            pred = model(X['data'][:, :91, :].to(device))
            preds[model_file].append(pred.cpu().detach().numpy())

        pickle.dump(preds, open(model_file.replace('.ckpt', '_test_preds.p'), 'wb'))

        # trainer = Trainer()
        # test_result = trainer.test(model, loader_test)
        #
        # model_preds = test_result[0]['model_output']
        # pickle.dump(model_preds, open(os.path.join(model_path, f'pred_test_{model_file}.p'), 'w'))
        # del test_result[0]['model_output']
        #
        # with open(os.path.join(f'{model_path}', f'model_performance_{model_file}.json'),
        #           'w') as fp:
        #     json.dump(test_result[0], fp)

