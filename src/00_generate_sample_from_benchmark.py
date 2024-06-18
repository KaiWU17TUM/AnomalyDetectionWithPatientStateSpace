import os
import pickle

import h5py
import hdf5plugin

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask.dataframe as dd
from IPython.display import display, HTML

# from tableone import TableOne

os.chdir('../')
from utils.config_dataset import *

import warnings
warnings.filterwarnings("ignore")

from utils.ClassDataset import BenchmarkAEDataset
from torch.utils.data import DataLoader


if __name__=='__main__':
    # ############################################################
    # # VALID PATIENT IDS
    # ############################################################
    # general_ext = pd.read_parquet('hirid_benchmark/general_table_extended.parquet')
    # pid_valid = general_ext['patientid'].unique()
    # print(f"Number of unique patient IDs: {len(pid_valid)}")
    #
    # # remove patient with no discharge location
    # pid_nodist = general_ext.loc[pd.isnull(general_ext['discharge_status']), 'patientid'].tolist()
    # pid_valid = [pid for pid in pid_valid if pid not in pid_nodist]
    # print(f"Remove patient with no discharge status... --- {len(pid_valid)}")
    #
    # # remove patient with no APACHE group
    # pid_noapache = general_ext.loc[(pd.isnull(general_ext['APACHE II Group'])) & (pd.isnull(general_ext['APACHE IV Group'])), 'patientid'].tolist()
    # pid_valid = [pid for pid in pid_valid if pid not in pid_noapache]
    # print(f"Remove patient with no APACHE group... --- {len(pid_valid)}")
    #
    # # merge APACHE II and APACHE IV, remove patient with multiple
    # APACHE_DICT = {
    #     v: k for k in APACHE_BENCHMARK_MERGE for v in APACHE_BENCHMARK_MERGE[k]
    # }
    #
    # pid_multiapache = []
    # apache_dict_merged = {}
    # for pid in pid_valid:
    #     apache2 = general_ext.loc[general_ext['patientid'] == pid, 'APACHE II Group'].item()
    #     apache4 = general_ext.loc[general_ext['patientid'] == pid, 'APACHE IV Group'].item()
    #     if ~np.isnan(apache2):
    #         apache_merge = APACHE_DICT[int(apache2)]
    #         apache_dict_merged[pid] = [apache_merge]
    #     if ~np.isnan(apache4):
    #         apache_merge = APACHE_DICT[int(apache4)]
    #         if pid in apache_dict_merged:
    #             if apache_merge in apache_dict_merged[pid]:
    #                 continue
    #             else:
    #                 apache_dict_merged[pid] += [apache_merge]
    #                 pid_multiapache += [pid]
    #         else:
    #             apache_dict_merged[pid] = [apache_merge]
    #
    # pid_valid = [pid for pid in pid_valid if pid not in pid_multiapache]
    # print(f"Remove patient with multiple APACHE groups... --- {len(pid_valid)}")
    # pickle.dump(pid_valid, open('processed-benchmark/pid_valid.p', 'wb'))
    #
    # ############################################################
    # # LOAD BENCHMARK DATA AND SELECT RELEVANT COLUMNS
    # ############################################################
    # # select columns of patient data
    # var_ref = pd.read_csv('processed-v2/varref.tsv', sep='\t')
    # var_ref['variableid'] = var_ref['variableid'].astype(int)
    # path_merge = 'hirid_benchmark/merged_stage/'
    # df_merge = dd.read_parquet(path_merge)
    # path_common = 'hirid_benchmark/common_stage/'
    # df_common = pd.read_parquet(path_common)
    # df_common = df_common[df_common['patientid'].isin(pid_valid)]
    #
    # cols = df_merge.columns
    # id_obs = [int(col[2:]) for col in df_merge.columns if 'vm' in col]
    # id_med = [int(col[2:]) for col in df_merge.columns if 'pm' in col]
    #
    # # selected pharma IDs
    # pharma_path = 'physionet.org/files/hirid/1.1.1/raw_stage'
    # pharma_data = pd.read_parquet(os.path.join(pharma_path, 'pharma_records', 'parquet'))
    # pharmaids = {}
    #
    # metaids_selected_med = []
    # for vid in INPUT_OF_INTEREST:
    #     metaid = var_ref.loc[var_ref['variableid'] == vid, 'metavariableid'].item()
    #     if metaid in metaids_selected_med:
    #         continue
    #     metaids_selected_med.append(metaid)
    #
    # for metaid in metaids_selected_med:
    #     pharmaids[metaid] = []
    #     for vid in var_ref.loc[var_ref['metavariableid'] == metaid, 'variableid']:
    #         if vid not in pharmaids[metaid]:
    #             pharmaids[metaid].append(vid)
    #
    # col_obs = [var_ref.loc[var_ref['metavariableid'] == vid, 'metavariablename'].unique().item() for vid in id_obs]
    # col_med = [var_ref.loc[var_ref['metavariableid'] == vid, 'metavariablename'].unique().item() for vid in id_med]
    # col_med_selected = var_ref.loc[
    #     var_ref['metavariableid'].isin(metaids_selected_med), 'metavariablename'].unique().tolist()
    # col_info = [col for col in df_common.columns if col not in col_obs + col_med]
    #
    # patient_data = df_common[col_info + col_med_selected + col_obs]
    # pickle.dump(patient_data, open('processed-benchmark/patient_data.p', 'wb'))
    #
    #
    # # patient information with LOS and merged APACHE
    # patient_info = general_ext[general_ext['patientid'].isin(pid_valid)]
    # patient_info['APACHE MERGED'] = None
    # patient_info['LOS'] = None
    #
    # for pid in tqdm(pid_valid):
    #     assert len(apache_dict_merged[pid]) == 1
    #     patient_info.loc[patient_info['patientid'] == pid, 'APACHE MERGED'] = apache_dict_merged[pid][0]
    #
    #     los = patient_data[patient_data['patientid'] == pid].shape[0] / 12 / 24
    #     patient_info.loc[patient_info['patientid'] == pid, 'LOS'] = los
    # pickle.dump(patient_info, open('processed-benchmark/patient_info.p', 'wb'))

    pid_valid = pickle.load(open('processed-benchmark/pid_valid.p', 'rb'))
    patient_info = pickle.load(open('processed-benchmark/patient_info.p', 'rb'))
    patient_data = pickle.load(open('processed-benchmark/patient_data.p', 'rb'))
    patient_data = patient_data[patient_data.columns[:127]]

    # index sample ids for training AE
    print("Assigning sampld IDs...")
    # patient_data['sampleid_6h'] = -1
    # for pid in tqdm(pid_valid):
    #     los = patient_info.loc[patient_info['patientid']==pid, 'LOS'].item() * 24 * 60
    #     # data = patient_data.loc[patient_data['patientid']==pid]
    #     # mask = (patient_data['patientid']==pid) & (patient_data['datetime']<los//SAMPLE_LEN_AE*SAMPLE_LEN_AE)
    #     patient_data.loc[(patient_data['patientid'] == pid) & (patient_data['datetime'] < los // SAMPLE_LEN_AE * SAMPLE_LEN_AE), 'sampleid_6h'] \
    #         = patient_data.loc[(patient_data['patientid']==pid) & (patient_data['datetime']<los//SAMPLE_LEN_AE*SAMPLE_LEN_AE)].apply(
    #         lambda row: row.datetime // SAMPLE_LEN_AE,
    #         axis=1
    #     )
    #     # check = patient_data.loc[patient_data['patientid']==pid, ['datetime', 'sampleid_6h']]
    # pickle.dump(patient_data, open('processed-benchmark/patient_data.p', 'wb'))



    # patient_data.rename(columns={'sampleid_6h': 'sampleid_6h_local'}, inplace=True)
    patientids = list(patient_data['patientid'])
    sampleids = list(patient_data['sampleid_6h_local'])
    newblock = True
    id_new = 1
    ids_new = []
    for i, (pid, id) in enumerate(tqdm(zip(patientids, sampleids))):
        if id == -1:
            ids_new.append(-1)
            continue
        if newblock:
            newblock = False

        ids_new.append(id_new)

        if i + 1 < len(sampleids):
            pid_next = patientids[i + 1]
            id_next = sampleids[i + 1]
            if (pid_next != pid) or (id_next != id):
                newblock = True
                id_new += 1

    patient_data['sampleid_6h'] = ids_new
    patient_data = pd.merge(patient_data,
                                 patient_info[['patientid', 'discharge_status', 'APACHE MERGED', 'LOS']],
                                 on=['patientid'])
    pickle.dump(patient_data, open('processed-benchmark/patient_data.p', 'wb'))

    # pid_valid = pickle.load(open('processed-benchmark/pid_valid.p', 'rb'))
    # patient_info = pickle.load(open('processed-benchmark/patient_info.p', 'rb'))
    # patient_data = pickle.load(open('processed-benchmark/patient_data.p', 'rb'))

    # data_statistics = patient_data.describe(percentiles=[.001, .01, .05, .1, .25, .5, .75, .9, .95, .99, .999])
    # data_statistics.to_csv('processed-benchmark/patient_data_statistics.csv')
    # data_statistics = pd.read_csv('processed-benchmark/patient_data_statistics.csv', header=[0], index_col=[0])

    # columns = ['age', 'sex', 'height', 'LOS', 'APACHE MERGED', 'discharge_status'] + PHYSIO_BENCHMARK + MED_BENCHMARK
    # categorical = ['sex', 'discharge_status', 'APACHE MERGED']
    # groupby = ['discharge_status']
    # summary_table = TableOne(patient_data, columns=columns, categorical=categorical, groupby=groupby, pval=False)
    # print(summary_table.tabulate(tablefmt="fancy_grid"))
    # summary_table.to_excel('processed-benchmark/statistics.xlsx')

    # # add endpoint columns & remaining LOS
    # path_endpoints = 'hirid_benchmark/endpoints'
    # df_endpoints = pd.read_parquet(path_endpoints)
    #
    # df_endpoints['resp_failure_status_level'] = 0
    # df_endpoints.loc[df_endpoints['resp_failure_status_relabel']==True]

    #
    # dataset = BenchmarkAEDataset(patient_data, norm=True, selected_physio=True)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    #
    # for sample in loader:
    #     info = sample['info']
    #     data = sample['data']







    print(111)


