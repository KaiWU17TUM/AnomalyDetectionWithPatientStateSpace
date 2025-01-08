import pickle
from pathlib import Path
import time

import numpy as np
import torch

from sklearn.model_selection import train_test_split

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.config_dataset import *
from utils.data_io import load_train_test_dataset
from utils.ClassMonoModel import MED_ITERATIVE_MONO

RANDOMSEED=2024
torch.manual_seed(RANDOMSEED)
np.random.seed(RANDOMSEED)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_physio_constraints(n_feat, n_feat_med, n_info_emb, idx_feat=[4]):
    # physio
    # input: current vital + pred vital + current med + accumulated med + embedded info
    # output: pred_vital
    mono_physio_curr = - np.eye(n_feat)
    mono_physio_pred = np.zeros_like(mono_physio_curr)
    mono_vaso = [
        [0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0],
    ]
    if n_feat != 7:
        mono_vaso = np.array(mono_vaso)[:, idx_feat]
    # mono_med_other = [[0] * n_feat] * (n_feat_med-3)
    mono_info = [[0] * n_feat] * n_info_emb

    # constraints = np.concatenate((mono_physio_curr, mono_physio_pred, mono_vaso, mono_med_other, mono_vaso, mono_med_other, mono_info), axis=0)
    constraints = np.concatenate((mono_physio_curr, mono_physio_pred, mono_vaso, mono_vaso, mono_info), axis=0)

    print(f"MONO CONSTRAINT PHYSIO: {constraints.shape}")

    return constraints

if __name__ == '__main__':
    base_path = 'processed-merge-v3/'
    model_type = 'MED_ITERATIVE_MONO'

    device = 'cuda'
    batchsize = 16
    lr = 5e-4
    #input
    seq_len = 180
    n_feat = 1
    n_feat_med = 3
    n_emb_info = 8
    dropout = 0.1
    # mono param
    regression_type = 'fc'        # 'mono' or 'fc'
    n_emb_mono = 35
    n_groupsort = 5
    # Prior knowledge of vasoactive agents on vital signs
    # medications: 'norepinephrine', 'epinephrine', 'dobutamine',
    # vital signs: 'HR', 'RR', 'SpO2', 'ABPd', 'ABPm', 'ABPs', 'ZVD',
    physio_constraints = get_physio_constraints(n_feat, n_feat_med, n_emb_info)
    # iterative steps
    n_step = 7
    n_step_med = 15
    # loss weight
    alpha = 5
    beta = 8

    model_name = f"{regression_type}-{n_feat}input-{alpha}-{beta}-{n_step}step-{n_step_med}-nedstep-{dropout}dropout-{batchsize}-{lr}"
    print(model_type, model_name)

    DATA = load_train_test_dataset(
        batchsize=batchsize,
        sample_dict_file='sample_dict_vasopressor_filtered80.p',
        type='vaso',
        RANDOMSEED=RANDOMSEED, norm=True, smooth=True, interpolate=True,
        n_step=n_step, n_step_med=n_step_med,
        data_path=base_path
    )
    pid_valid = DATA['pid_valid']
    norm_params = DATA['norm_params']
    norm_params_info = DATA['norm_params_info']
    patient_info = DATA['patient_info']
    dataset_train = DATA['dataset_train']
    dataset_val = DATA['dataset_val']
    dataset_test = DATA['dataset_test']
    loader_train = DATA['loader_train']
    loader_val = DATA['loader_val']
    loader_test = DATA['loader_test']

    # Train model
    config = {
        "device": device,
        "batchsize": batchsize,
        "lr": lr,
        "seq_len": seq_len,
        "n_step": n_step,
        "n_step_med": n_step_med,
        "n_feat": n_feat,
        "n_feat_med": n_feat_med,
        "n_emb_info": n_emb_info,
        "dropout": dropout,
        'regression_type': regression_type,
        # MONO
        'n_emb_mono': n_emb_mono,
        'n_groupsort': n_groupsort,
        'monotonic_constraints_physio': physio_constraints,
        # loss
        'alpha': alpha,
        'beta': beta,
    }

    if model_type == 'MED_ITERATIVE_MONO':
        model = MED_ITERATIVE_MONO(config).to(device)
    else:
        print(f'{model_type} is not supported!')

    model_save_path = f'models/{model_type}/{model_name}'
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    model_version = os.listdir(model_save_path)

    version = 0
    versions = [int(v.split('_')[1]) for v in model_version if 'version_' in v]
    if len(versions) > 0:
        version = sorted(versions)[-1] + 1

    callbacks = [
        ModelCheckpoint(
            monitor='val_loss_ar',
            mode='min',
            save_top_k=1,
            dirpath=f'{model_save_path}/version_{version}',
            filename='epoch{epoch:02d}-val_loss{val_loss:.5f}',
            auto_insert_metric_name=False
        ),
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            dirpath=f'{model_save_path}/version_{version}',
            filename='epoch{epoch:02d}-val_loss_pred{val_loss_pred:.5f}',
            auto_insert_metric_name=False
        ),
        ModelCheckpoint(
            monitor='val_cstr',
            mode='min',
            save_top_k=1,
            dirpath=f'{model_save_path}/version_{version}',
            filename='epoch{epoch:02d}-val_loss_pred{val_loss_pred:.5f}',
            auto_insert_metric_name=False
        ),
        EarlyStopping(
            monitor='val_loss_ar',
            mode='min',
            patience=30,
        )
    ]

    logger = TensorBoardLogger(
        f'{model_save_path}',
        name=model_name,
        version=version)

    trainer = Trainer(
        max_epochs=250,
        gpus=1,
        callbacks=callbacks,
        logger=logger,
        # resume_from_checkpoint=os.path.join(checkpoint_dir, "checkpoint"),
    )
    train_time_start = time.time()
    f'{model_save_path}/version_{version}'
    trainer.fit(model, train_dataloaders=loader_train, val_dataloaders=loader_val)
    pickle.dump(config, open(f'{model_save_path}/version_{version}/model_config.p', 'wb'))
    train_time_total = time.time() - train_time_start

    with open(f'{model_save_path}/train_time.txt', 'a') as train_time_file:
        train_time_file.write(f'{model_name}: {train_time_total}\n')

