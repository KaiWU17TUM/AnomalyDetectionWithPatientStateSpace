from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from sklearn.model_selection import train_test_split

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.config_dataset import *
from utils.ClassDataset import CusDataset
from utils.ClassAE import LSTM_AE

RANDOMSEED = 2024
torch.manual_seed(RANDOMSEED)
np.random.seed(RANDOMSEED)




if __name__ == '__main__':
    archive = pickle.load(open(os.path.join(path_processed, 'training_data_injectiononly.p'), 'rb'))
    samples = archive['samples_norm']
    sample_dict= archive['sample_dict']
    idx_train= archive['idx_train']
    idx_test= archive['idx_test']
    pid_train = archive['pid_train']
    pid_test = archive['pid_test']
    del archive

    patient_info = pickle.load(open(os.path.join(path_processed, 'patient_info_valid.p'), 'rb'))
    dischargestatus = [patient_info.loc[patient_info['patientid']==pid, 'discharge_status'].item()
                       for pid in pid_train]
    pid_tr, pid_val = train_test_split(pid_train, test_size=0.2, random_state=RANDOMSEED, stratify=dischargestatus)
    idx_tr = [i for i in sample_dict if sample_dict[i]['pid'] in pid_tr]
    idx_val = [i for i in sample_dict if sample_dict[i]['pid'] in pid_val]

    X_train = samples[idx_tr]
    X_val = samples[idx_val]
    X_test = samples[idx_test]
    dataset_train = CusDataset(X_train)
    dataset_val = CusDataset(X_val)
    dataset_test = CusDataset(X_test)
    loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=16)
    loader_val = DataLoader(dataset_val, batch_size=64, shuffle=True, num_workers=16)
    loader_test = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=16)


    device = 'cuda'
    seq_len = 181
    n_feat = 12
    n_emb = 128
    n_layer = 2
    dropout = 0.0

    config = {
        'device': device,
        "seq_len": seq_len,
        "n_feat": n_feat,
        "n_emb": n_emb,
        'n_layer': n_layer,
        "lr": 5e-3,
        "dropout": dropout,
    }

    model_type = 'LSTM-AE'
    model_name = f"{n_layer}layer-{n_emb}hidden-{dropout}dropout"
    model_save_path = f'models/{model_name}'
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
            save_top_k=3,
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

    model = LSTM_AE(config).to(device)

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
    trainer.fit(model, train_dataloaders=loader_train, val_dataloaders=loader_val)
    train_time_total = time.time() - train_time_start

    with open(f'{model_save_path}/train_time.txt', 'a') as train_time_file:
        train_time_file.write(f'{model_name}: {train_time_total}\n')

    # dataset = CusDataset(X_test)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    #
    # for i, x in enumerate(loader):
    #     d = x['data']
    #     dosage = x['dosage']
    #     mask = x['med_mask']
    #     print(i)

    # patient_info = pickle.load(open(os.path.join(path_processed, 'patient_info_valid.p'), 'rb'))
    # dischargestatus = [patient_info.loc[patient_info['patientid']==pid, 'discharge_status'].item()
    #                    for pid in pid_train]
    #
    # # cross validation
    # for i in range(5):
    #     pid_tr, pid_val= train_test_split(pid_train, test_size=0.2, random_state=i, stratify=dischargestatus)

