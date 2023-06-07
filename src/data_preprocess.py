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
    #####################################################
    # dataset files
    #####################################################
    data_raw_path = '../physionet.org/files/hirid/1.1.1/raw_stage/observation_tables_parquet'
    data_merged_path = '../physionet.org/files/hirid/1.1.1/merged_stage/merged_stage_parquet'
    data_imputed_path = '../physionet.org/files/hirid/1.1.1/imputed_stage/imputed_stage_parquet'

    patient_info_file = '../physionet.org/files/hirid/1.1.1/reference_data/general_table.csv'
    variable_file = '../physionet.org/files/hirid/1.1.1/reference_data/hirid_variable_reference.csv'
    apache_group_file = '../processed/APACHE_groups.csv'

    variable_table = pd.read_csv(variable_file)

    # process selected pharma and physio data dict
    selected_pharma_file = '../processed/HiRID_selected_variables-input_dict.csv'
    selected_pharma = pd.read_csv(selected_pharma_file)
    selected_physio_file = '../processed/HiRID_selected_variables-output_dict.csv'
    selected_physio = pd.read_csv(selected_physio_file)

    datasheet = pd.read_csv('../processed/helper_physio_dict_preprocess.csv')
    selected_physio_final = pd.read_csv('../processed/HiRID_selected_variables-output_dict_to_be_processed.csv')

    # delete variables with no occurences or unclear meaning
    selected_physio_final = selected_physio_final.drop(
        selected_physio_final[~selected_physio_final['variableid'].isin(datasheet['variableid'])].index
    ).reset_index(drop=True)

    # label variebles which are set by medical staff
    selected_physio_final['isSetting'] = 0
    selected_physio_final.loc[
        selected_physio_final['variableid'].isin(datasheet.loc[datasheet['isSetting']==1, 'variableid']),
        'isSetting'
    ]=1

    for vid in datasheet['variableid'].values:
        selected_physio_final.loc[selected_physio_final['variableid']==vid, 'type'] = \
            datasheet.loc[datasheet['variableid']==vid, 'type'].item()
        # Shape: B-bell, H-halfBell, M-mixGaussian, D-discrete
        selected_physio_final.loc[selected_physio_final['variableid']==vid, 'shape'] = \
            datasheet.loc[datasheet['variableid']==vid, 'shape'].item()
        selected_physio_final.loc[selected_physio_final['variableid']==vid, 'unit'] = \
            variable_table.loc[(variable_table['Source Table']=='Observation')
                               & (variable_table['ID']==vid), 'Unit'].item()

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

    # assign unique ID
    uid = 1
    for vid in tqdm(selected_physio_final['variableid'].values):
        dup = False
        for k in merge_items:
    #         print(k)
            for vlist in merge_items[k]:
    #             print(vlist)
                if vid in vlist:
                    dup = True
                    tmp = selected_physio_final[selected_physio_final['variableid'].isin(vlist)]
    #                 print(tmp[['variableid', 'uid']])
                    if tmp['uid'].notnull().sum() == 0:
                        selected_physio_final.loc[selected_physio_final['variableid']==vid, 'uid'] = uid
                        uid += 1
    #                     print(f'{vid}: {uid}')
                        continue
                    else:
                        uid_exist = tmp.loc[tmp['uid'].notnull(), 'uid'].unique().item()
                        selected_physio_final.loc[selected_physio_final['variableid']==vid, 'uid'] = uid_exist
    #                     print(selected_physio_final[selected_physio_final['variableid']==vid])
                        continue
    #     print(f'after for loop: {vid}, {uid}')
        if not dup:
            selected_physio_final.loc[selected_physio_final['variableid']==vid, 'uid'] = uid
            uid += 1
    selected_physio_final['uid'] = selected_physio_final['uid'].astype(int)

    selected_physio_final.to_csv('../processed/HiRID_selected_variables-output_processed.csv')


    print('Loading data...')
    # merge APACHE II & IV categories
    apache_table = pd.read_csv(apache_group_file)
    apache_table = apache_table.drop(apache_table[apache_table['II'] == 'IV'].index)
    apache_table['II'] = apache_table['II'].astype(int)
    apache_dict = {}
    for apache_name in apache_table['Name'].unique():
        apache_dict[apache_name] = apache_table.loc[apache_table['Name']==apache_name, 'II'].values.tolist()

    # load patient data
    patient_info = pd.read_csv(patient_info_file)
    patient_info['admissiontime'] = pd.to_datetime(patient_info['admissiontime'])
    patient_info['los'] = -1

    pharma_path = '../physionet.org/files/hirid/1.1.1/raw_stage/pharma_records_parquet'
    pharma_data = pd.read_parquet(os.path.join(pharma_path, 'pharma_records', 'parquet'))
    patient_data = pd.read_parquet(os.path.join(data_raw_path, 'observation_tables', 'parquet'))

    # patient with entries of selected pharma
    pid_with_selected_pharma = pharma_data.loc[pharma_data['pharmaid'].isin(selected_pharma['variableid']), 'patientid'].astype(int).unique().tolist()
    pickle.dump(pid_with_selected_pharma, open('../processed/pid_with_selected_pharma.p', 'wb'))

    #####################################################
    # Grouping patient with APACHE
    #####################################################
    print('Grouping patient with APACHE...')
    pid_apache_group = {}
    for apache in tqdm(apache_dict):
        patient_apache = patient_data.loc[(patient_data['variableid'].isin([9990002, 9990004])) & (patient_data['value'].isin(apache_dict[apache]))]
        pid_apache_group[apache] = patient_apache['patientid'].unique()
    pickle.dump(pid_apache_group, open('../processed/pid_apache_group.p', 'wb'))

    pid_apache_with_selected_pharma = {}
    for apache in tqdm(apache_dict):
        pid_apache_with_selected_pharma[apache] = list(set(pid_apache_group[apache]).intersection(set(pid_with_selected_pharma)))
    pickle.dump(pid_apache_with_selected_pharma, open('../processed/pid_apache_group_with_selected_pharma.p', 'wb'))
    print('Grouping patient with APACHE...finished')

    # remove pid with multiple apache entries
    pid_valid = []
    for apache in pid_apache_with_selected_pharma:
        pid_valid += pid_apache_with_selected_pharma[apache]
    print(f'PIDs with selected pharma: {len(pid_valid)}')
    pid_duplicates = [item for item, count in collections.Counter(pid_valid).items() if count > 1]
    pid_valid = set(pid_valid)
    print(f'Unique PIDs with selected pharma: {len(pid_valid)}')
    pid_valid = [item for item in pid_valid if item not in pid_duplicates]
    print(f'PIDs with selected pharma and one single APACHE group: {len(pid_valid)}')

    # dataset with selected pids and variables
    pharma_data_selected = pharma_data[(pharma_data['patientid'].isin(pid_valid))
                                       & (pharma_data['pharmaid'].isin(selected_pharma['variableid']))]
    patient_data_selected = patient_data.loc[(patient_data['patientid'].isin(pid_valid))
                                             & (patient_data['variableid'].isin(selected_physio['variableid']))]

    # remove pid without discharge status
    for patientid in patient_info[patient_info['discharge_status'].isnull()]['patientid'].unique():
        try:
            pid_valid.remove(patientid)
        except:
            pass
    print(f'PIDs - removing unknown discharge status: {len(pid_valid)}')

    # remove pid with NAN data entries
    for patientid in patient_data_selected.loc[patient_data_selected['value'].isnull()]['patientid'].unique():
        try:
            pid_valid.remove(patientid)
        except:
            pass
    print(f'PIDs - removing NAN patient data entries: {len(pid_valid)}')

    # pid APACHE group with valid pids
    count = []
    pid_group_valid = {}
    for apache in pid_apache_with_selected_pharma:
        pid_group_valid[apache] = list(set(pid_apache_with_selected_pharma[apache]).intersection(set(pid_valid)))
        count += pid_group_valid[apache]
    print('Valid PID number: ', len(set(count)))

    patient_info_selected = patient_info[patient_info['patientid'].isin(pid_valid)]
    patient_info_selected['APACHE'] = 'None'
    for pid in tqdm(pid_valid):
        pdata = patient_data_selected[patient_data_selected['patientid'] == pid]
        los = pdata['datetime'].max() - patient_info_selected.loc[
            patient_info_selected['patientid'] == pid, 'admissiontime'].item()
        patient_info_selected.loc[patient_info_selected['patientid'] == pid, 'los'] = los
        for apache in pid_group_valid:
            if pid in pid_group_valid[apache]:
                patient_info_selected.loc[patient_info_selected['patientid']==pid, 'APACHE'] = apache

    patient_info_valid = patient_info_selected[patient_info_selected['patientid'].isin(pid_valid)].reset_index(drop=True)
    pharma_data_valid = pharma_data_selected[pharma_data_selected['patientid'].isin(pid_valid)].reset_index(drop=True)
    patient_data_valid = patient_data_selected[patient_data_selected['patientid'].isin(pid_valid)].reset_index(drop=True)

    # remove prelimitary lab values
    patient_data_valid = patient_data_valid.drop(patient_data_valid[patient_data_valid['type']=='P'].index).reset_index(drop=True)

    print(f"Patient info: {patient_info_valid['patientid'].unique().shape}, {patient_info_valid.shape}")
    print(f"Pharma data: {pharma_data_valid['patientid'].unique().shape}, {pharma_data_valid.shape}")
    print(f"Physio data: {patient_data_valid['patientid'].unique().shape}, {patient_data_valid.shape}")

    pickle.dump(pid_valid, open('../processed/pid_valid.p', 'wb'))
    pickle.dump(pid_group_valid, open('../processed/pid_group_valid.p', 'wb'))
    pickle.dump(patient_info_valid, open('../processed/patient_info_valid.p', 'wb'))
    pickle.dump(pharma_data_valid, open('../processed/pharma_data_valid.p', 'wb'))
    pickle.dump(patient_data_valid, open('../processed/patient_data_valid.p', 'wb'))