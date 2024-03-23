from pathlib import Path
import time
import glob
import json

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
from utils.ClassPredictor import PRETRAINED_PRED_ALLMED
from utils.ClassAE import GuidedLSTM_AE_ALLMED, LSTM_AE_ALLMED

RANDOMSEED=2024
torch.manual_seed(RANDOMSEED)
np.random.seed(RANDOMSEED)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


if __name__ == '__main__':
    # model_type = 'GuidedLSTM_PRED_PRETRAINED_ALLMED'
    model_type = 'LSTM_PRED_PRETRAINED_ALLMED'
    kfolds = 5
    dropout_pred = 0.5
    lr_pred = 1e-3


    if model_type == 'GuidedLSTM_PRED_PRETRAINED_ALLMED':
        ModelClass = GuidedLSTM_AE_ALLMED
    elif model_type =='LSTM_PRED_PRETRAINED_ALLMED':
        ModelClass = LSTM_AE_ALLMED
    else:
        print(f'{model_type} is not supported!')


    pretrained_paths = {
        'GuidedLSTM_PRED_PRETRAINED_ALLMED': [
            'models/GuidedLSTM-AE-ALLMED/2layer-128hidden-0.1dropout-64-0.001/version_0',
            'models/GuidedLSTM-AE-ALLMED/2layer-128hidden-0.2dropout-64-0.001/version_0',
            'models/GuidedLSTM-AE-ALLMED/2layer-256hidden-0.1dropout-64-0.001/version_0',
            'models/GuidedLSTM-AE-ALLMED/3layer-128hidden-0.1dropout-64-0.001/version_0',
        ],
        'LSTM_PRED_PRETRAINED_ALLMED': [
            'models/LSTM-AE-ALLMED/2layer-128hidden-0.2dropout-64-0.001/version_0',
            'models/LSTM-AE-ALLMED/2layer-128hidden-0.1dropout-64-0.001/version_0',
            'models/LSTM-AE-ALLMED/2layer-256hidden-0.1dropout-64-0.001/version_0',
            'models/LSTM-AE-ALLMED/3layer-128hidden-0.1dropout-64-0.001/version_0',
        ]
    }

    archive = pickle.load(open(os.path.join(path_processed, 'training_data_allmed.p'), 'rb'))
    samples = archive['samples_norm']
    sample_dict = archive['sample_dict']
    idx_train = archive['idx_train']
    idx_test = archive['idx_test']
    pid_train = archive['pid_train']
    pid_test = archive['pid_test']
    del archive

    device = 'cuda'
    seq_len = 181
    n_feat = 10

    for model_pretrained_path in pretrained_paths[model_type]:
        params = model_pretrained_path.split('/')[2]
        lr = float(params.split('-')[4])
        batchsize = int(params.split('-')[3])
        n_emb = int(params.split('-')[1].strip('hidden'))
        n_layer = int(params.split('-')[0].strip('layer'))
        dropout = float(params.split('-')[2].strip('dropout'))
        config = {
            'device': device,
            "seq_len": seq_len,
            "n_feat": n_feat,
            "n_emb": n_emb,
            'n_layer': n_layer,
            "lr": lr,
            "dropout": dropout,
            "dropout_pred": dropout_pred,
            "lr_pred": lr_pred,
        }


        model_name = f"{dropout_pred}-{lr_pred}-{n_layer}layer-{n_emb}hidden-{dropout}dropout-{batchsize}-{lr}"
        print(model_type, model_name)

        model_save_path = f'models/{model_type}/{model_name}'
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
        model_version = os.listdir(model_save_path)

        version = 0
        versions = [int(v.split('_')[1]) for v in model_version if 'version_' in v]
        if len(versions) > 0:
            version = sorted(versions)[-1] + 1

        kf = KFold(n_splits=kfolds, shuffle=True, random_state=RANDOMSEED)
        kf_splits = [k for k in kf.split(idx_train)]

        for kfold in range(kfolds):
            idx_tr, idx_val = kf_splits[kfold]

            X_train = samples[idx_tr]
            X_val = samples[idx_val]
            # X_test = samples[idx_test]
            dataset_train = CusDataset(X_train)
            dataset_val = CusDataset(X_val)
            # dataset_test = CusDataset(X_test)
            loader_train = DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=16)
            loader_val = DataLoader(dataset_val, batch_size=batchsize, shuffle=False, num_workers=16)
            # loader_test = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=16)

            model_pretrained_k = glob.glob(f'{model_pretrained_path}/cv{kfold}*.ckpt')
            if len(model_pretrained_k) != 1:
                raise ValueError(f"{len(model_pretrained_k)} pretrained model found. DUPLICATED!")
            print(model_pretrained_k)
            model_pretrained = ModelClass.load_from_checkpoint(checkpoint_path=model_pretrained_k[0],
                                                               config=config)
            model = PRETRAINED_PRED_ALLMED(config, model_pretrained)


            callbacks = [
                ModelCheckpoint(
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1,
                    dirpath=f'{model_save_path}/version_{version}',
                    filename=f'cv{kfold}'+'epoch{epoch:02d}-val_loss{val_loss:.5f}',
                    auto_insert_metric_name=False
                ),
                EarlyStopping(
                    monitor='val_loss',
                    mode='min',
                    patience=15,
                )
            ]

            logger = TensorBoardLogger(
                f'{model_save_path}',
                name=f'cv{kfold}' + model_name,
                version=version)

            trainer = Trainer(
                max_epochs=250,
                gpus=1,
                callbacks=callbacks,
                logger=logger,
                # resume_from_checkpoint=os.path.join(checkpoint_dir, "checkpoint"),
            )
            train_time_start = time.time()
            trainer.fit(model, train_dataloaders=loader_train, val_dataloaders=loader_val)
            pickle.dump(config, open(f'{model_save_path}/version_{version}/model_config.p', 'wb'))
            train_time_total = time.time() - train_time_start

            with open(f'{model_save_path}/train_time.txt', 'a') as train_time_file:
                train_time_file.write(f'{model_name}: {train_time_total}\n')

            summary = ModelSummary(model, max_depth=-1)
            print(summary)

            try:
                ckpts = glob.glob(os.path.join(f'{model_save_path}/version_{version}', '*val_loss*.ckpt'))
                roc_score = [float(os.path.splitext(ckpt)[0].split('val_loss')[1]) for ckpt in ckpts]
                ckpt_best = ckpts[np.argmin(roc_score)]
                test_result = trainer.test(ckpt_path=ckpt_best)
                with open(os.path.join(f'{model_save_path}/version_{version}', f'model_performance_test_data_cv{kfold}.json'), 'w') as fp:
                    json.dump(test_result[0], fp)
            except:
                pass

        try:
            with open(os.path.join(f'{model_save_path}/version_{version}', 'pretrained_version.txt'), 'a') as f:
                f.write(f'{model_pretrained_path}')
        except:
            pass