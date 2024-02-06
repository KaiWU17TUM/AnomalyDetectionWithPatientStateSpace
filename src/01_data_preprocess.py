import os
import pickle
import collections

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import pandas as pd
import numpy as np


if __name__=='__main__':
    #####################################################
    # dataset files
    #####################################################
    # original data from HIRID
    data_raw_path = '../physionet.org/files/hirid/1.1.1/raw_stage/observation_tables_parquet'
    data_merged_path = '../physionet.org/files/hirid/1.1.1/merged_stage/merged_stage_parquet'
    data_imputed_path = '../physionet.org/files/hirid/1.1.1/imputed_stage/imputed_stage_parquet'
    # original data reference table from HIRID
    patient_info_file = '../physionet.org/files/hirid/1.1.1/reference_data/general_table.csv'
    variable_file = '../physionet.org/files/hirid/1.1.1/reference_data/hirid_variable_reference_v1.csv'
    apache_group_file = '../physionet.org/files/hirid/1.1.1/reference_data/APACHE_groups.csv'
    variable_table = pd.read_csv(variable_file)

    # process selected pharma and physio data dict
    path_processed = '../processed-v2/'
    selected_pharma_file = os.path.join(path_processed, 'HiRID_selected_variables-input_dict_v2.csv')
    selected_pharma = pd.read_csv(selected_pharma_file)
    selected_physio_file = os.path.join(path_processed, 'HiRID_selected_variables-output_dict_v2.csv')
    selected_physio = pd.read_csv(selected_physio_file)

    datasheet = pd.read_csv(os.path.join(path_processed, 'helper_physio_dict_preprocess_v2.csv'))


    # delete variables with no occurences or unclear meaning
    # 5685 - Respiratory rate: (43601686, 8)                    --- extreme large values regarding respiration rate, not sure if the values make sense.
    # 24000520 - Potassium[Moles / volume] in Blood: (0, 8)     --- no recording in entire dataset
    # 24000519 - Sodium[Moles / volume] in Blood: (0, 8)        --- no recording in entire dataset
    # 24000658 - Sodium[Moles / volume] in Blood: (0, 8)        --- no recording in entire dataset
    # 24000573 - Creatinine[Moles / volume] in Urine: (0, 8)    --- no recording in entire dataset
    selected_physio = selected_physio.drop(
        selected_physio[~selected_physio['variableid'].isin(datasheet['variableid'])].index
    ).reset_index(drop=True)

    # label variebles which are set by medical staff
    selected_physio['isSetting'] = 0
    selected_physio.loc[
        selected_physio['variableid'].isin(datasheet.loc[datasheet['isSetting']==1, 'variableid']),
        'isSetting'
    ]=1
    # copy data type / unit information
    for vid in datasheet['variableid'].values:
        try:
            selected_physio.loc[selected_physio['variableid']==vid, 'type'] = \
                datasheet.loc[datasheet['variableid']==vid, 'type'].item()
            # Shape: B-bell, H-halfBell, M-mixGaussian, D-discrete
            selected_physio.loc[selected_physio['variableid']==vid, 'shape'] = \
                datasheet.loc[datasheet['variableid']==vid, 'shape'].item()
            selected_physio.loc[selected_physio['variableid']==vid, 'unit'] = \
                variable_table.loc[(variable_table['Source Table']=='Observation')
                                   & (variable_table['ID']==vid) & (variable_table['ID']!=15001166), 'Unit'].item()
        except:
            print(111)

    # duplicated variables to be merged or added up
    dup_names = [item for item, count in collections.Counter(selected_physio['variablename']).items() if count > 1]
    duplicates = selected_physio[selected_physio['variablename'].isin(dup_names)].groupby('variablename')

    # assign unique ID
    uid = 1
    for vid in tqdm(selected_physio['variableid'].values):
        dup = False
        for dname in dup_names:
            vlist = duplicates.get_group(dname)['variableid'].tolist()
#             print(vlist)
            if vid in vlist:
                dup = True
                tmp = selected_physio[selected_physio['variableid'].isin(vlist)]
#                 print(tmp[['variableid', 'uid']])
                if tmp['uid'].notnull().sum() == 0:
                    selected_physio.loc[selected_physio['variableid']==vid, 'uid'] = uid
                    uid += 1
#                     print(f'{vid}: {uid}')
                    continue
                else:
                    uid_exist = tmp.loc[tmp['uid'].notnull(), 'uid'].unique().item()
                    selected_physio.loc[selected_physio['variableid']==vid, 'uid'] = uid_exist
#                     print(selected_physio[selected_physio['variableid']==vid])
                    continue
    #     print(f'after for loop: {vid}, {uid}')
        if not dup:
            selected_physio.loc[selected_physio['variableid']==vid, 'uid'] = uid
            uid += 1
    selected_physio['uid'] = selected_physio['uid'].astype(int)

    selected_physio.to_csv(os.path.join(path_processed, 'selected_physio.csv'))

    #####################################################
    # Load raw data from HiRID dataset
    #####################################################
    print('Loading data...')
    # load patient information
    patient_info = pd.read_csv(patient_info_file)
    patient_info['admissiontime'] = pd.to_datetime(patient_info['admissiontime'])
    patient_info['los'] = -1
    # load pharma data
    pharma_path = '../physionet.org/files/hirid/1.1.1/raw_stage/pharma_records_parquet'
    pharma_data = pd.read_parquet(os.path.join(pharma_path, 'pharma_records', 'parquet'))
    # load physio data
    patient_data = pd.read_parquet(os.path.join(data_raw_path, 'observation_tables', 'parquet'))
    # patient ids with entries of selected pharma
    pid_with_selected_pharma = pharma_data.loc[pharma_data['pharmaid'].isin(selected_pharma['variableid']), 'patientid'].astype(int).unique().tolist()
    pickle.dump(pid_with_selected_pharma, open(os.path.join(path_processed, 'pid_with_selected_pharma.p'), 'wb'))


    #####################################################
    # Patient APACHE groups
    #####################################################
    print('Grouping patient with APACHE...')
    # merge APACHE II & IV categories
    apache_table = pd.read_csv(apache_group_file)
    apache_table = apache_table.drop(apache_table[apache_table['II'] == 'IV'].index)
    apache_table['II'] = apache_table['II'].astype(int)
    apache_dict = {}
    for apache_name in apache_table['Name'].unique():
        apache_dict[apache_name] = apache_table.loc[apache_table['Name'] == apache_name, 'II'].values.tolist()
    print('Grouping patient with APACHE...')
    pid_apache_group = {}
    for apache in tqdm(apache_dict):
        patient_apache = patient_data.loc[(patient_data['variableid'].isin([9990002, 9990004])) & (patient_data['value'].isin(apache_dict[apache]))]
        pid_apache_group[apache] = patient_apache['patientid'].unique().tolist()
    print('Grouping patient with APACHE...finished')

    # remove pid with multiple apache entries
    pid_valid = []
    for apache in pid_apache_group:
        pid_valid += pid_apache_group[apache]
    print(f'PIDs with selected pharma: {len(pid_valid)}')
    pid_duplicates = [item for item, count in collections.Counter(pid_valid).items() if count > 1]
    pid_valid = set(pid_valid)
    print(f'Unique PIDs with selected pharma: {len(pid_valid)}')
    pid_valid = [item for item in pid_valid if item not in pid_duplicates]
    print(f'PIDs with selected pharma and one single APACHE group: {len(pid_valid)}')

    # # dataset with selected pids and variables
    # pharma_data_selected = pharma_data[(pharma_data['patientid'].isin(pid_valid))
    #                                    & (pharma_data['pharmaid'].isin(selected_pharma['variableid']))]
    # patient_data_selected = patient_data.loc[(patient_data['patientid'].isin(pid_valid))
    #                                          & (patient_data['variableid'].isin(selected_physio['variableid']))]

    # remove pid without discharge status
    for patientid in patient_info[patient_info['discharge_status'].isnull()]['patientid'].unique():
        try:
            pid_valid.remove(patientid)
        except:
            pass
    print(f'PIDs - removing unknown discharge status: {len(pid_valid)}')

    # # remove pid with NAN data entries
    # for patientid in patient_data.loc[patient_data['value'].isnull()]['patientid'].unique():
    #     try:
    #         pid_valid.remove(patientid)
    #     except:
    #         pass
    # print(f'PIDs - removing NAN patient data entries: {len(pid_valid)}')

    # pid APACHE group with valid pids
    count = []
    pid_group_valid = {}
    for apache in pid_apache_group:
        pid_group_valid[apache] = list(set(pid_apache_group[apache]).intersection(set(pid_valid)))
        count += pid_group_valid[apache]
    print('Valid PID number: ', len(set(count)))

    patient_info_selected = patient_info[patient_info['patientid'].isin(pid_valid)]
    patient_info_selected['APACHE'] = 'None'
    for pid in tqdm(pid_valid):
        pdata = patient_data[patient_data['patientid'] == pid]
        los = pdata['datetime'].max() - patient_info_selected.loc[
            patient_info_selected['patientid'] == pid, 'admissiontime'].item()
        patient_info_selected.loc[patient_info_selected['patientid'] == pid, 'los'] = los
        for apache in pid_group_valid:
            if pid in pid_group_valid[apache]:
                patient_info_selected.loc[patient_info_selected['patientid']==pid, 'APACHE'] = apache

    patient_info_valid = patient_info_selected[patient_info_selected['patientid'].isin(pid_valid)].reset_index(drop=True)

    pharma_data_valid = pharma_data[(pharma_data['patientid'].isin(pid_valid))
                                    & (pharma_data['pharmaid'].isin(selected_pharma['variableid']))].reset_index(drop=True)
    patient_data_valid = patient_data.loc[(patient_data['patientid'].isin(pid_valid))
                                          & (patient_data['variableid'].isin(selected_physio['variableid']))].reset_index(drop=True)

    # remove preliminary lab values
    patient_data_valid = patient_data_valid.drop(patient_data_valid[patient_data_valid['type']=='P'].index).reset_index(drop=True)

    print(f"Patient info: {patient_info_valid['patientid'].unique().shape}, {patient_info_valid.shape}")
    print(f"Pharma data: {pharma_data_valid['patientid'].unique().shape}, {pharma_data_valid.shape}")
    print(f"Physio data: {patient_data_valid['patientid'].unique().shape}, {patient_data_valid.shape}")

    pickle.dump(pid_valid, open(os.path.join(path_processed, 'pid_valid.p'), 'wb'))
    pickle.dump(pid_group_valid, open(os.path.join(path_processed, 'pid_group_valid.p'), 'wb'))
    pickle.dump(patient_info_valid, open(os.path.join(path_processed, 'patient_info_valid.p'), 'wb'))
    pickle.dump(pharma_data_valid, open(os.path.join(path_processed, 'pharma_data_valid.p'), 'wb'))
    pickle.dump(patient_data_valid, open(os.path.join(path_processed, 'patient_data_valid.p'), 'wb'))

    patient_data_valid['uid'] = patient_data_valid['variableid'].map(dict(zip(selected_physio['variableid'], selected_physio['uid'])))
    pickle.dump(patient_data_valid, open(os.path.join(path_processed, 'patient_data_valid_with_uid.p'), 'wb'))

    # delete pharma entries that do not exist in the whole dataset
    selected_pharma = selected_pharma.drop(
        selected_pharma[~selected_pharma['variableid'].isin(pharma_data_valid['pharmaid'].unique())].index
    ).reset_index(drop=True)
    pickle.dump(selected_pharma, open(os.path.join(path_processed, 'selected_pharma.p'), 'wb'))

    # pharma_data_valid = pickle.load(open(os.path.join(path_processed, 'pharma_data_valid.p'), 'rb'))
    # patient_data_valid = pickle.load(open(os.path.join(path_processed, 'patient_data_valid_with_uid.p'), 'rb'))
    # selected_physio = pd.read_csv(os.path.join(path_processed, 'selected_physio.csv'))

    # calculate data percentiles for later normalization
    # physio data
    cols = ['physioname', 'min', 0.1, 0.5, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9, 'max']
    idx = selected_physio['uid'].unique().tolist()
    norm_param_physio = pd.DataFrame(columns=cols, index=idx)
    norm_param_physio['physioname'] = selected_physio.groupby('uid').first()['variablename']
    for uid in tqdm(selected_physio['uid'].unique()):
        name = selected_physio.loc[selected_physio['uid'] == uid, 'variablename'].unique().item()
        data = patient_data_valid.loc[(patient_data_valid['uid'] == uid)]
        percs = [np.nanpercentile(data['value'].values, p) for p in cols[2:13]]
        norm_param_physio.loc[uid, cols[2:13]] = percs
        norm_param_physio.loc[uid, cols[1]] = data['value'].min()
        norm_param_physio.loc[uid, cols[13]] = data['value'].max()
    norm_param_physio.to_csv(os.path.join(path_processed, 'robnorm_pararms_physio.csv'))
    # pharma data
    cols = ['pharmaname', 'min', 0.1, 0.5, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9, 'max']
    # idx = selected_pharma['variableid'].tolist()
    # norm_param_pharma = pd.DataFrame(columns=cols, index=idx)
    norm_param_pharma = pd.DataFrame(columns=cols)
    # norm_param_pharma['pharmaname'] = selected_pharma['variablename'].tolist()
    for pharma in tqdm(selected_pharma['variableid']):
        try:
            name = selected_pharma.loc[selected_pharma['variableid'] == pharma, 'variablename'].unique().item().replace(';', ',')
            data = pharma_data_valid.loc[(pharma_data_valid['pharmaid'] == pharma) & (pharma_data_valid['givendose'] != 0)]
            if data.shape[0] == 0:
                continue
            # norm_param_pharma = norm_param_pharma.reindex(index=list(norm_param_pharma.index) + [pharma])
            percs = [np.percentile(data['givendose'].values, p) for p in cols[2:13]]
            norm_param_pharma.loc[pharma, 'pharmaname'] = name
            norm_param_pharma.loc[pharma, cols[2:13]] = percs
            norm_param_pharma.loc[pharma, cols[1]] = data['givendose'].min()
            norm_param_pharma.loc[pharma, cols[13]] = data['givendose'].max()
        except:
            print(f'{pharma}: {name}')
    norm_param_pharma.to_csv(os.path.join(path_processed, 'robnorm_pararms_pharma.csv'))
