from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.config_dataset import *
from utils.ClassDataset import BenchmarkAEDataset
from utils.ClassPredictor import GuidedLSTM_PRED_BENCHMARK

RANDOMSEED=2024
torch.manual_seed(RANDOMSEED)
np.random.seed(RANDOMSEED)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


if __name__ == '__main__':
    model_type = 'GuidedLSTM_PRED_BENCHMARK'

    device = 'cuda'
    seq_len = 72
    n_feat = 10
    n_feat_med = 7
    n_emb = 128
    n_layer = 2
    dropout = 0.1
    lr = 1e-3
    batchsize = 64

    model_name = f"{n_layer}layer-{n_emb}hidden-{dropout}dropout-{batchsize}-{lr}"
    print(model_type, model_name)

    # load data
    print("Loading dataset")
    pid_valid = pickle.load(open('processed-benchmark/pid_valid.p', 'rb'))
    patient_info = pickle.load(open('processed-benchmark/patient_info.p', 'rb'))
    patient_data = pickle.load(open('processed-benchmark/patient_data.p', 'rb'))
    data_statistics = pd.read_csv('processed-benchmark/patient_data_statistics.csv', header=[0], index_col=[0])
    dischargestatus = [patient_info.loc[patient_info['patientid'] == pid, 'discharge_status'].item()
                       for pid in pid_valid]
    pid_train, pid_test = train_test_split(pid_valid, test_size=0.2, random_state=RANDOMSEED, stratify=dischargestatus)
    pid_tr, pid_val = train_test_split(pid_train, test_size=0.2, random_state=RANDOMSEED,
                                       stratify=[ds for (pid, ds) in zip(pid_valid, dischargestatus) if
                                                 pid in pid_train])
    dataset_train = BenchmarkAEDataset(patient_data[patient_data['patientid'].isin(pid_tr)], norm=True,
                                       selected_physio=True)
    dataset_val = BenchmarkAEDataset(patient_data[patient_data['patientid'].isin(pid_val)], norm=True,
                                     selected_physio=True)
    dataset_test = BenchmarkAEDataset(patient_data[patient_data['patientid'].isin(pid_test)], norm=True,
                                      selected_physio=True)
    loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=32)
    loader_val = DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=32)
    loader_test = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=16)


    # Train model
    config = {
        'device': device,
        "seq_len": seq_len,
        "n_feat": n_feat,
        "n_feat_med": n_feat_med,
        "n_emb": n_emb,
        'n_layer': n_layer,
        "lr": lr,
        "dropout": dropout,
    }

    if model_type == 'GuidedLSTM_PRED_BENCHMARK':
        model = GuidedLSTM_PRED_BENCHMARK(config).to(device)
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
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=15,
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