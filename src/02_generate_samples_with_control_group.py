import os
import pickle
from pathlib import Path
from datetime import timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.dask import TqdmCallback
import dask.dataframe as dd
from sklearn.model_selection import train_test_split

os.chdir('/home/kai/DigitalICU/Experiments/HIRID-PatientStateSpace/')
from utils.config_dataset import *
from utils.preprocess_benchmark import drop_duplicates_pharma

import warnings
warnings.filterwarnings("ignore")

RANDOMSEED=2024

if __name__ == '__main__':
    save_path = 'processed-merge/'
    save_path_merged_data_per_pat = os.path.join(save_path, 'merged_data_per_pat')
    pid_valid = pickle.load(open(os.path.join(save_path, 'pid_valid_00.p'), 'rb'))
    pharma_data = pickle.load(open(os.path.join(save_path, 'pharma_data_00.p'), 'rb'))

    ############################################################
    # Generate sample index -- medication
    ############################################################
    THRES_MIN = 6
    THRES_BEFORE_MED = 3
    THRES_AFTER_MED = 3
    sample_dict = {med: {} for med in MED_BENCHMARK}

    # idx_sample = {med: 0 for med in MED_BENCHMARK}
    # for pid in tqdm(pid_valid):
    #     df = pickle.load(open(os.path.join(save_path_merged_data_per_pat, f"{pid}.p"), 'rb'))
    #     df[MED_BENCHMARK] = df[MED_BENCHMARK].fillna(0)
    #     for med in MED_BENCHMARK:
    #         med_data = df[med]
    #         med_data = med_data[med_data > 0]
    #         if med_data.shape[0] > 0:
    #             med_start = med_data.index[0]
    #             if med_start - df.index[0] >= timedelta(hours=THRES_MIN) \
    #                     and df.index[-1] - med_start >= timedelta(hours=THRES_AFTER_MED):
    #                 sample_start = med_start - timedelta(hours=THRES_BEFORE_MED)
    #                 sample_end = med_start + timedelta(hours=THRES_AFTER_MED)
    #                 sample_dict[med][idx_sample[med]] = (pid, sample_start, sample_end)
    #                 idx_sample[med] += 1
    # pickle.dump(sample_dict, open(os.path.join(save_path, 'sample_lookuptable_per_med.p'), 'wb'))

    for med in sample_dict:
        print(f"{med:<10} --- {len(sample_dict[med].keys())}")
    sample_dict_vaso = pickle.load(open(os.path.join(save_path, 'sample_lookuptable_per_med.p'), 'rb'))

    ############################################################
    # Generate sample index -- control group with no vasoactive agents
    ############################################################
    sample_dict_control = {}
    MED_VASO = MED_BENCHMARK[:3]
    idx = 0
    for pid in tqdm(pid_valid):
        df = pickle.load(open(os.path.join(save_path_merged_data_per_pat, f"{pid}.p"), 'rb'))
        df[MED_BENCHMARK] = df[MED_BENCHMARK].fillna(0)

        t_start = df.index[0]
        t_end = t_start + timedelta(hours=THRES_MIN)
        t_last = df.index[-1]
        while t_last - t_end >= timedelta(hours=THRES_MIN):
            check = df.loc[(df.index>=t_start) & (df.index<t_end), MED_VASO].sum(axis=1)
            if check.sum() == 0:
                sample_dict_control[idx] = (pid, t_start, t_end)
                idx += 1
                t_start = t_end
            else:
                t_med_last = check.index[check>0][-1]
                if t_med_last == check.index[0]:
                    t_start = check.index[1]
                else:
                    t_start = t_med_last
            t_end = t_start + timedelta(hours=THRES_MIN)

    pickle.dump(sample_dict_control, open(os.path.join(save_path, 'sample_lookuptable_control_group.p'), 'wb'))



