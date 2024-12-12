import pickle
from pathlib import Path
import time

import torch

from sklearn.model_selection import train_test_split

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.config_dataset import *
from utils.data_io import load_train_test_dataset
from utils.ClassMonoModel import MED_PHYSIO_MONO

RANDOMSEED=2024
torch.manual_seed(RANDOMSEED)
np.random.seed(RANDOMSEED)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

if __name__ == '__main__':
    base_path = 'processed-merge-v3/'
    model_type = 'MED_PHYSIO_MONO'

    device = 'cuda'
    batchsize = 8
    lr = 5e-4
    seq_len = 180
    n_feat = 7
    n_feat_med = 7
    encoder_type = 'TCN'    # 'CNN' / 'TCN'
    n_emb = 168             # CNN: 196 / TCN: 168
    n_emb_info = 16
    dropout = 0.1
    n_emb_mono = 35
    n_groupsort = 5
    # pretrained AE model with the control samples
    model_path_ae = 'models/AE_PHYSIO/TCN-168hidden-0.1dropout-32-0.0005/version_0/'
    # Prior knowledge of vasoactive agents on vital signs
    # medications: 'norepinephrine', 'epinephrine', 'dobutamine',
    # vital signs: 'HR', 'RR', 'SpO2', 'ABPd', 'ABPm', 'ABPs', 'ZVD',
    monotonic_constraints = [
        [0] * n_feat
    ] * n_emb + [
        [0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0],
    ] + [
        [0] * n_feat
    ] * (n_feat_med - 3)
    monotonic_constraints = np.array(monotonic_constraints)
    print(f"MONO CONSTRAINT PHYSIO: {monotonic_constraints.shape}")

    #     [
    #     [0] * n_emb + [0, 0, 0, 1, 1, 1, 1],
    #     [0] * n_emb + [1, 1, 1, 1, 1, 1, 1],
    #     [0] * n_emb + [1, 0, 0, 0, 0, 0, 0],
    #     [0] * (n_emb + n_feat_med),
    #     [0] * (n_emb + n_feat_med),
    #     [0] * (n_emb + n_feat_med),
    #     [0] * (n_emb + n_feat_med),
    # ]
    # monotonic_constraints = np.array(monotonic_constraints).T

    model_name = f"{encoder_type}-{n_emb}hidden-{dropout}dropout-{batchsize}-{lr}"
    print(model_type, model_name)

    DATA = load_train_test_dataset(
        batchsize=batchsize,
        sample_dict_file='sample_dict_vasopressor_filtered80.p',
        type='vaso',
        RANDOMSEED=RANDOMSEED, norm=True, smooth=True, interpolate=True,
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
        "n_feat": n_feat,
        "n_feat_med": n_feat_med,
        "encoder_type": encoder_type,
        "n_emb": n_emb,
        "n_emb_info": n_emb_info,
        "dropout": dropout,
        # AE model
        "model_path_ae": model_path_ae,
        # MONO
        'n_emb_mono': n_emb_mono,
        'n_groupsort': n_groupsort,
        'monotonic_constraints': monotonic_constraints,
    }

    if model_type == 'MED_PHYSIO_MONO':
        model = MED_PHYSIO_MONO(config).to(device)
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
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            dirpath=f'{model_save_path}/version_{version}',
            filename='epoch{epoch:02d}-val_loss{val_loss:.5f}',
            auto_insert_metric_name=False
        ),
        ModelCheckpoint(
            monitor='val_loss_pred',
            mode='min',
            save_top_k=1,
            dirpath=f'{model_save_path}/version_{version}',
            filename='epoch{epoch:02d}-val_loss_pred{val_loss_pred:.5f}',
            auto_insert_metric_name=False
        ),
        EarlyStopping(
            monitor='val_loss',
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

