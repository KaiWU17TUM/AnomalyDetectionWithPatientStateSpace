import os
import pickle
import collections
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import probscale
import seaborn

from IPython.display import display, HTML

if __name__=='__main__':
    selected_pharma_file = '../processed/HiRID_selected_variables-input_dict.csv'
    selected_pharma = pd.read_csv(selected_pharma_file)

    selected_physio = pd.read_csv('../processed/HiRID_selected_variables-output_processed.csv')

    uid_dict = {}
    for index, row in selected_physio.iterrows():
        uid_dict[row['variableid']] = row['uid']

    pid_list = pickle.load(open('../processed/pid_valid.p', 'rb'))
    pid_group = pickle.load(open('../processed/pid_group_valid.p', 'rb'))
    patient_info = pickle.load(open('../processed/patient_info_valid.p', 'rb'))
    pharma_data = pickle.load(open('../processed/pharma_data_valid.p', 'rb'))
    patient_data = pickle.load(open('../processed/patient_data_valid.p', 'rb'))
    df_physio = patient_data.copy(deep=True)
    df_physio['uid'] = 0
    for vid in tqdm(selected_physio['variableid'].unique()):
        uid = selected_physio.loc[selected_physio['variableid'] == vid, 'uid'].item()
        df_physio.loc[df_physio['variableid'] == vid, 'uid'] = uid

    selected_pharma = selected_pharma.drop(
        selected_pharma[~selected_pharma['variable_id'].isin(pharma_data['pharmaid'].unique())].index
    ).reset_index(drop=True)

