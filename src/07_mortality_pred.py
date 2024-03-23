import pickle
from pathlib import Path
import time
import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchsampler import ImbalancedDatasetSampler

from utils.config_dataset import *
from utils.ClassDataset import ClassifierDataset, CusDataset
from utils.ClassEncoderClassifier import PretrainedAE, GuidedMortalityClassification_ALLMED
from utils.ClassAE import GuidedLSTM_AE_ALLMED

RANDOMSEED=2024
torch.manual_seed(RANDOMSEED)
np.random.seed(RANDOMSEED)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':
    model_type = 'GuidedMortalityClassification_ALLMED'
    # model_type = 'GuidedLOS_PRED_ALLMED'
    include_med = False

    if model_type == 'GuidedMortalityClassification_ALLMED':
        ModelClass = GuidedMortalityClassification_ALLMED
    else:
        print(f'{model_type} is not supported!')

    device = 'cuda'
    seq_len = 181
    n_feat = 10
    batchsize = 64

    ######################## LOAD DATA ########################
    archive = pickle.load(open(os.path.join(path_processed, 'training_data_allmed.p'), 'rb'))
    samples = archive['samples_norm']
    sample_dict= archive['sample_dict']
    idx_train= archive['idx_train']
    idx_test= archive['idx_test']
    pid_train = archive['pid_train']
    pid_test = archive['pid_test']
    del archive
    labels = pickle.load(open(os.path.join(path_processed, 'training_label_allmed_normed.p'), 'rb'))
    labels['survived'] = list(1 - np.array(labels['survived']))

    print(sum(labels['survived']))

    #################################################################
    ## LOAD PRETRAINED AUTOENCODER
    ## GENERATE ENCODED DATA AND OUTPUT FROM THE AUTOENCODER
    #################################################################
    # model_pretrained_path = 'models/GuidedLSTM-AE-ALLMED/3layer-128hidden-0.1dropout-64-0.001/version_0'
    # params = model_pretrained_path.split('/')[2]
    # lr = float(params.split('-')[4])
    # batchsize = int(params.split('-')[3])
    # n_emb = int(params.split('-')[1].strip('hidden'))
    # n_layer = int(params.split('-')[0].strip('layer'))
    # dropout = float(params.split('-')[2].strip('dropout'))
    # config_pretrained = {
    #     'device': device,
    #     "seq_len": seq_len,
    #     "n_feat": n_feat,
    #     "n_emb": n_emb,
    #     'n_layer': n_layer,
    #     "lr": 1e-2,
    #     "dropout": dropout,
    # }
    #
    # X_emb = {i: [] for i in range(5)}
    # X_reconstruct = {i: [] for i in range(5)}
    # dataset = CusDataset(samples)
    # loader = DataLoader(dataset,  batch_size=128, shuffle=False, num_workers=32)
    # for i, file in enumerate(glob.glob(f'{model_pretrained_path}/*.ckpt')):
    #     print(f'KFOLD {i}')
    #     model_pretrained = GuidedLSTM_AE_ALLMED.load_from_checkpoint(checkpoint_path=file, config=config_pretrained)
    #     model = PretrainedAE(config_pretrained, model_pretrained)
    #     model.to(device)
    #     model.eval()
    #     for X in tqdm(loader):
    #         x_rec, x_emb = model(X['data'].to(device))
    #         X_reconstruct[i].append(x_rec.cpu().detach().numpy())
    #         X_emb[i].append(x_emb.cpu().detach().numpy())
    # pickle.dump((X_reconstruct, X_emb),
    #             open(os.path.join(path_processed, 'traning_data_encoded_with_pretrained_GAE.p'), 'wb'))
    # for i in range(5):
    #     X_reconstruct[i] = np.concatenate(X_reconstruct[i])
    #     X_emb[i] = np.concatenate(X_emb[i])
    # X_rec = (X_reconstruct[0] + X_reconstruct[1] + X_reconstruct[2] + X_reconstruct[3] + X_reconstruct[4]) / 5
    # for i in range(5):
    #     X_emb[i] = X_emb[i][:, None, :]
    # X_enc = np.concatenate((X_emb[0], X_emb[1], X_emb[2], X_emb[3], X_emb[4]), axis=1)
    # pickle.dump((X_rec, X_enc), open(os.path.join(path_processed, 'traning_data_classification.p'), 'wb'))

    X_rec, X_enc = pickle.load(open(os.path.join(path_processed, 'traning_data_classification.p'), 'rb'))




    patient_info = pickle.load(open(os.path.join(path_processed, 'patient_info_valid.p'), 'rb'))
    dischargestatus = [patient_info.loc[patient_info['patientid'] == pid, 'discharge_status'].item()
                       for pid in pid_train]
    pid_tr, pid_val = train_test_split(pid_train, test_size=0.2, random_state=RANDOMSEED, stratify=dischargestatus)
    idx_tr = [i for i in sample_dict if sample_dict[i]['pid'] in pid_tr]
    idx_val = [i for i in sample_dict if sample_dict[i]['pid'] in pid_val]

    for k in labels:
        labels[k] = np.array(labels[k])
    label_train = {k: labels[k][idx_tr] for k in labels}
    label_val = {k: labels[k][idx_val] for k in labels}
    label_test = {k: labels[k][idx_test] for k in labels}

    X_train = (samples[idx_tr], X_rec[idx_tr], X_enc[idx_tr])
    X_val = (samples[idx_val], X_rec[idx_val], X_enc[idx_val])
    # X_test = (X_rec[idx_test], X_enc[idx_test])
    dataset_train = ClassifierDataset(X_train, label_train)
    dataset_val = ClassifierDataset(X_val, label_val)
    # dataset_test = ClassifierDataset(X_test, label_test)
    loader_train = DataLoader(dataset_train, batch_size=batchsize, num_workers=16, sampler=ImbalancedDatasetSampler(dataset_train))
    loader_val = DataLoader(dataset_val, batch_size=batchsize, shuffle=False, num_workers=16)
    # loader_test = DataLoader(dataset_test, batch_size=batchsize, shuffle=False, num_workers=16)



    dropout = 0.7
    lr = 2e-4

    config = {
        'device': device,
        "seq_len": seq_len,
        "n_feat": n_feat,
        "lr": lr,
        "dropout": dropout,
    }

    model = ModelClass(config)

    model_name = f"{dropout}dropout-{batchsize}-{lr}"
    print(model_type, model_name)
    model_save_path = f'models/{model_type}/{model_name}'
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    model_version = os.listdir(model_save_path)

    version = 0
    versions = [int(v.split('_')[1]) for v in model_version if 'version_' in v]
    if len(versions) > 0:
        version = sorted(versions)[-1] + 1

    callbacks = [
        ModelCheckpoint(
            monitor='val_roc',
            mode='max',
            save_top_k=1,
            dirpath=f'{model_save_path}/version_{version}',
            filename='epoch{epoch:02d}-val_roc{val_roc:.5f}',
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
    trainer.fit(model, train_dataloaders=loader_train, val_dataloaders=loader_val)
    pickle.dump(config, open(f'{model_save_path}/version_{version}/model_config.p', 'wb'))
    train_time_total = time.time() - train_time_start

    with open(f'{model_save_path}/train_time.txt', 'a') as train_time_file:
        train_time_file.write(f'{model_name}: {train_time_total}\n')