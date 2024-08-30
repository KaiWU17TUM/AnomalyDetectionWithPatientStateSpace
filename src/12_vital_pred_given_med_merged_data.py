import pickle
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
from utils.ClassDataset import MergedDataset
from utils.ClassSOM import SOM_CLF, SOM_PRED, SOM_MTL
from utils.data_io import load_train_test_dataset

RANDOMSEED=2024
torch.manual_seed(RANDOMSEED)
np.random.seed(RANDOMSEED)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


if __name__ == '__main__':
    model_type = 'SOM_PRED'

    device = 'cuda'
    batchsize = 8
    lr = 5e-4
    seq_len = 180
    n_feat = 7
    n_feat_med = 3
    encoder_type = 'CNN'
    n_emb = 196
    n_emb_info =16
    n_emb_med =196
    som_size = 8
    r_neighbor = 3
    dropout = 0.1
    alpha = 3
    beta = 1


    model_name = f"{encoder_type}-{n_emb}hidden-{som_size}som-{r_neighbor}r-{dropout}dropout-{batchsize}-{lr}"
    print(model_type, model_name)

    DATA = load_train_test_dataset(batchsize=batchsize, RANDOMSEED=RANDOMSEED, norm=True, smooth=True)
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


    # load data
    # print("Loading dataset")
    # data_path = 'processed-merge/'
    # pid_valid = pickle.load(open(os.path.join(data_path, 'pid_valid_00.p'), 'rb'))
    # patient_info = pickle.load(open(os.path.join(data_path, 'patient_info.p'), 'rb'))
    # sample_dict = pickle.load(open(os.path.join(data_path, 'sample_dict_vasopressor.p'), 'rb'))
    # norm_params = pickle.load(open('processed-merge/norm_params_vasopressor.p', 'rb'))
    # norm_params_info = pickle.load(open('processed-merge/norm_params_info_vasopressor.p', 'rb'))
    #
    # dischargestatus = []
    # for i in sample_dict:
    #     status = patient_info.loc[patient_info['patientid'] == sample_dict[i][1], 'discharge_status'].item()
    #     label = 1 if status == 'dead' else 0
    #     dischargestatus.append(label)
    # dischargestatus = np.array(dischargestatus)
    #
    # selected_physio = ['HR', 'RR', 'SpO2', 'ABPd', 'ABPm', 'ABPs', 'ZVD']
    # selected_med = ['norepinephrine', 'epinephrine', 'dobutamine']
    # train_test_indices = pickle.load(open(os.path.join(data_path, 'train_test_split_vasopressor.p'), 'rb'))
    # sampleid_train, sampleid_test = train_test_indices['train'], train_test_indices['test']
    # sampleid_tr, sampleid_val = train_test_split(sampleid_train, test_size=0.2, random_state=RANDOMSEED,
    #                                    stratify=dischargestatus[sampleid_train])
    #
    # dataset_train = MergedDataset(sample_dict={item[0]: item[1] for item in sample_dict.items() if item[0] in sampleid_tr},
    #                               df_info=patient_info, norm=True, selected_physio=selected_physio,selected_med=selected_med)
    # dataset_val = MergedDataset(sample_dict={item[0]: item[1] for item in sample_dict.items() if item[0] in sampleid_val},
    #                               df_info=patient_info, norm=True, selected_physio=selected_physio,selected_med=selected_med)
    # dataset_test = MergedDataset(sample_dict={item[0]: item[1] for item in sample_dict.items() if item[0] in sampleid_test},
    #                               df_info=patient_info, norm=True, selected_physio=selected_physio,selected_med=selected_med)
    # loader_train = DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=32)
    # loader_val = DataLoader(dataset_val, batch_size=batchsize, shuffle=False, num_workers=32)
    # loader_test = DataLoader(dataset_test, batch_size=batchsize, shuffle=False, num_workers=16)

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
        "n_emb_med": n_emb_med,
        "som_size": som_size,
        "r_neighbor": r_neighbor,
        "dropout": dropout,
        'alpha': alpha,
        'beta': beta,
    }

    if model_type == 'SOM_CLF':
        model = SOM_CLF(config).to(device)
    elif model_type == 'SOM_PRED':
        model = SOM_PRED(config).to(device)
    elif model_type =='SOM_MTL':
        model = SOM_MTL(config).to(device)
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