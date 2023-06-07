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


def percentile(n):
    def percentile_(x):
        return np.nanpercentile(x, n)

    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


# duplicated variables to be merged or added up
merge_items = {
    'merge': [
        [300, 310],
        [4000, 8280],
        [24000835, 24000866],
        [30005010, 30005110],
    ],
    'add': [
        [10010020, 10010070, 10010071, 10010072],
    ],
}

if __name__=='__main__':
    #####################################################
    # dataset files
    #####################################################
    data_raw_path = '../physionet.org/files/hirid/1.1.1/raw_stage/observation_tables_parquet'
    data_merged_path = '../physionet.org/files/hirid/1.1.1/merged_stage/merged_stage_parquet'
    data_imputed_path = '../physionet.org/files/hirid/1.1.1/imputed_stage/imputed_stage_parquet'

    patient_info_file = '../physionet.org/files/hirid/1.1.1/reference_data/general_table.csv'
    variable_file = '../physionet.org/files/hirid/1.1.1/reference_data/hirid_variable_reference.csv'
    apache_group_file = '../processed/apache_groups.csv'

    variable_table = pd.read_csv(variable_file)
    selected_pharma = pd.read_csv('../processed/HiRID_selected_variables-input_dict.csv')
    selected_physio = pd.read_csv('../processed/HiRID_selected_variables-output_processed.csv')

    uid_dict = {}
    for index, row in selected_physio.iterrows():
        uid_dict[row['variableid']] = row['uid']

    pid_list = pickle.load(open('../processed/pid_valid.p', 'rb'))
    pid_group = pickle.load(open('../processed/pid_group_valid.p', 'rb'))
    patient_info = pickle.load(open('../processed/patient_info_valid.p', 'rb'))
    pharma_data = pickle.load(open('../processed/pharma_data_valid.p', 'rb'))
    # patient_data_origin = pickle.load(open('../processed/patient_data_valid.p', 'rb'))
    # patient_data = patient_data_origin.copy(deep=True)
    # patient_data['uid'] = 0
    # for vid in tqdm(selected_physio['variableid'].unique()):
    #     uid = selected_physio.loc[selected_physio['variableid'] == vid, 'uid'].item()
    #     patient_data.loc[patient_data['variableid'] == vid, 'uid'] = uid
    # pickle.dump(patient_data, open('../processed/patient_data_valid_with_uid.p', 'wb'))
    patient_data = pickle.load(open('../processed/patient_data_valid_with_uid.p', 'rb'))

    selected_pharma = selected_pharma.drop(
        selected_pharma[~selected_pharma['variableid'].isin(pharma_data['pharmaid'].unique())].index
    ).reset_index(drop=True)

    COL_INFO_NUM = ['age', 'los']
    COL_INFO_CAT = ['sex', 'discharge_status', 'APACHE']
    COL_PHARMA = selected_pharma['variableid'].tolist()
    COL_PHYSIO_NUM = selected_physio.loc[selected_physio['type'] == 'n', 'variableid'].tolist()
    COL_PHYSIO_CAT = selected_physio.loc[selected_physio['type'] == 'c', 'variableid'].tolist()
    COL_PHYSIO_SETTING = selected_physio.loc[selected_physio['isSetting'] == 1, 'variableid'].tolist()
    # COL_PHYSIO_FLUID = selected_physio.loc[]

    # # ####################################################
    # # formated dataframe - patient information
    # # ####################################################
    # df_info = pd.DataFrame()
    # df_info = pd.concat((df_info, patient_info['patientid']), axis=1)
    # for col in COL_INFO_NUM:
    #     df_info = pd.concat((df_info, patient_info[col]), axis=1)
    # for col in COL_INFO_CAT:
    #     df_info = pd.concat((df_info, pd.get_dummies(patient_info[col], prefix=col)), axis=1)
    # df_info['los'] = df_info['los'].map(lambda x: x.days + x.seconds / 3600 / 24)
    # pickle.dump(df_info, open('../processed/df_info.p', 'wb'))


    # # ####################################################
    # # formated dataframe - pharma data
    # # ####################################################
    # df_pharma = pd.DataFrame(columns=['patientid'] + COL_PHARMA)
    # df_pharma['patientid'] = patient_info['patientid'].values
    # for pid in tqdm(pid_list):
    #     for pharma in COL_PHARMA:
    #         data = pharma_data.loc[(pharma_data['patientid'] == pid) & (pharma_data['pharmaid'] == pharma)]
    #         dosage = data['givendose'].sum()
    #         #         print(f"{pid} - {pharma}: {dosage}")
    #         df_pharma.loc[(df_pharma['patientid'] == pid), pharma] = dosage

    # # ####################################################
    # # formated dataframe - patient data
    # # ####################################################
    # Physio features for individual patients
    gb_pid_physioid = patient_data.groupby(['patientid', 'uid']).agg(
        {'value': ['median', 'mad',
                   percentile(1), percentile(5), percentile(25),
                   percentile(95), percentile(99),
                   ]
        }
    )
    # df_physio = pd.DataFrame()
    # df_physio = pd.concat((df_physio, patient_info['patientid']), axis=1)
    # for col in COL_PHYSIO_NUM:
    #     feat = pd.DataFrame
    #     df_physio = pd.concat((df_physio, feat), axis=1)
    #
    # for col in COL_PHYSIO_CAT:
    #     df_physio = pd.concat((df_physio, pd.get_dummies(patient_data[col], prefix=col)), axis=1)


    print(111)