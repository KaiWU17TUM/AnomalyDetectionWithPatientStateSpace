# patient LOS > 1 day


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
from utils.data_io import *
from utils.preprocess_benchmark import drop_duplicates_pharma

import warnings
warnings.filterwarnings("ignore")

RANDOMSEED=2024


def process_resp_endpoint(x):
    if x:
        if x == 'UNKNOWN':
            x = np.nan
        else:
            x = int(x.split('_')[1])
    else:
        x = np.nan
    return x

def process_and_save_pharma_data_per_patient(args):
    # format pharma data of selected patients into 2-min sampling freq.
    # calculate drug start time, end time and drug rate from the raw data
    pid = args['pid']
    pharma_pat = args['pharma_pat']
    pharmaref = args['pharmaref']
    save_path = args['save_path']

    idx_ts = pd.date_range(start=pharma_pat['givenat'].iloc[0], end=pharma_pat['givenat'].iloc[-1], freq='2T')
    df_formated = pd.DataFrame(0, columns=pharma_pat['pharmaid'].unique(), index=idx_ts)
    df_merged = pd.DataFrame(0, columns=MED_BENCHMARK, index=idx_ts)

    # calculate drug rate for injections using "pharmaactingperiod_min"
    df_inj = pharma_pat[pharma_pat['recordstatus'].isin(PHARMA_INJECTION)]
    # check for duplicated id for injection
    for iid in df_inj['infusionid'].unique():
        if df_inj[df_inj['infusionid']==iid].shape[0] != 1:
            for i, idx in enumerate(df_inj[df_inj['infusionid']==iid].index):
                try:
                    df_inj.loc[idx, 'infusionid'] = f"{iid}_000{i}"
                except:
                    print(pid, iid)
                    continue
    assert df_inj[df_inj['givendose'] <= 0].shape[0] == 0     # check for unclear dose
    df_inj.sort_values('givenat', inplace=True)
    # df_inj = df_inj.resample('2T', origin=df_inj.index[0])
    for iid in df_inj['infusionid'].unique():
        try:
            pharmaid = df_inj.loc[df_inj['infusionid']==iid, 'pharmaid'].item()
        except:
            print(pid, iid)
            continue
        dose = df_inj.loc[df_inj['infusionid']==iid, 'givendose'].item()
        acting_time = pharmaref.loc[pharmaref['pharmaid']==pharmaid, 'pharmaactingperiod_min'].item()
        convert_ratio = pharmaref.loc[pharmaref['pharmaid']==pharmaid, 'unitconversionfactor'].item()
        convert_ratio = 1 if np.isnan(convert_ratio) else convert_ratio
        rate = dose / acting_time * convert_ratio
        t_start = df_inj.loc[df_inj['infusionid']==iid, 'givenat'].item()
        t_end = t_start + timedelta(minutes=acting_time)
        df_formated.loc[(df_formated.index >= t_start) & (df_formated.index < t_end), pharmaid] += rate

    # calculate drug rate for infusions
    df_inf = pharma_pat[pharma_pat['recordstatus'].isin(PHARMA_INFUSION)]
    assert df_inf.loc[(df_inf['recordstatus'] == PHARMA_INFUSION_END) & (df_inf['cumulativedose'] == 0)].shape[0] == 0
    for iid in df_inf['infusionid'].unique():
        pharmaid = df_inf.loc[df_inf['infusionid'] == iid, 'pharmaid'].iloc[0]
        convert_ratio = pharmaref.loc[pharmaref['pharmaid'] == pharmaid, 'unitconversionfactor'].item()
        convert_ratio = 1 if np.isnan(convert_ratio) else convert_ratio

        df_inf_single = df_inf[df_inf['infusionid']==iid].copy()
        df_inf_single['rate'] = 0
        df_inf_single['rate'][:-1] = df_inf_single['givendose'].values[1:] \
                                     / (df_inf_single['givenat'].diff() / np.timedelta64(1, "m")).values[1:] * convert_ratio

        df_inf_single = df_inf_single.set_index('givenat').resample('2T', origin=df_formated.index[0]).mean()
        df_inf_single['rate'].fillna(method='ffill', inplace=True)
        df_formated.loc[df_inf_single.index, pharmaid] += df_inf_single['rate'].values

    if pd.isnull(df_formated).sum().sum() > 0:
        print('df_formated contains NAN: PID - ', pid)
    for med in df_merged.columns:
        metaid = pharmaref.loc[pharmaref['metavariablename']==med, 'metavariableid'].iloc[0]
        vids = MED_BENCHMARK_DICT_VID[metaid]
        cols = [col for col in df_formated.columns if col in vids]
        if len(cols) == 0:
            continue
        df_merged.loc[df_merged.index, med] = df_formated[cols].sum(axis=1)

    pickle.dump(df_merged, open(os.path.join(save_path, f"{pid}.p"), 'wb'))


def merge_and_save_data_per_pat(args):
    save_path = args['save_path']
    save_path_pharma_per_pat = args['save_path_pharma_per_pat']
    pid = args['pid']
    try:
        df_pharma = pickle.load(open(os.path.join(save_path_pharma_per_pat, f'{pid}.p'), 'rb'))
    except:
        df_pharma = pd.DataFrame(columns=MED_BENCHMARK)

    if df_pharma[MED_BENCHMARK[:3]].sum().sum() == 0:
        return

    # df_data = args['df_data']
    # df_endpoints = args['df_endpoints']
    df_data = patient_data_merge_stage[patient_data_merge_stage['patientid'] == pid]
    df_endpoint = df_endpoints[df_endpoints['patientid'] == pid]

    # df_data = args['df_data']
    # df_endpoint = args['df_endpoint']
    metaids_physio =args['metaids_physio']

    try:
        df_data = df_data[['datetime'] + [item[1] for item in metaids_physio.items()]]
        df_data.columns = ['datetime'] + [item[0] for item in metaids_physio.items()]
        df_data.set_index('datetime', inplace=True)
        df_data = df_data.resample('2T', origin=df_data.index[0]).mean()
        if df_pharma.shape[0] > 0:
            df_pharma = df_pharma.resample('2T', origin=df_data.index[0]).mean()
        if df_endpoint.shape[0] > 0:
            df_endpoint['resp_failure_status'] = df_endpoint['resp_failure_status'].apply(process_resp_endpoint)
            df_endpoint = df_endpoint[['datetime', 'resp_failure_status', 'circ_failure_status']].set_index('datetime')
            df_endpoint = df_endpoint.resample('2T', origin=df_data.index[0]).last()
            df_endpoint.fillna(method='ffill', inplace=True)
        df_all = df_data.join(df_pharma, how='outer')
        df_all = df_all.join(df_endpoint, how='outer')
        # remaining LOS
        los = (df_all.index[-1] - df_all.index).values / np.timedelta64(1, 'h') / 24
        df_all['LOS'] = los

        pickle.dump(df_all, open(os.path.join(save_path, f'{pid}.p'), 'wb'))

    except:
        with open(os.path.join(save_path, 'merge_data_failed.txt'), 'a+') as f:
            f.write(f'{pid},')
        print(f"Failed to generate merged data for patient: {pid}")



if __name__ == '__main__':
    general_ext = pd.read_parquet('hirid_benchmark/general_table_extended.parquet')
    # samples with vasoactive agents
    save_path = 'processed-merge-v3/'

    Path(save_path).mkdir(parents=True, exist_ok=True)

    ############################################################
    # VALID PATIENT IDS
    ############################################################
    print('Filtering patients with valid information...')
    general_ext = pd.read_parquet('hirid_benchmark/general_table_extended.parquet')
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


    ############################################################
    # APPEND APACHE & LOS TO PATIENT INFORMATION
    ############################################################
    # load patient data (merged stage)
    metaids_physio = {
        name: f"vm{varref.loc[varref['metavariablename'] == name, 'metavariableid'].unique().item()}" for name in
        PHYSIO_BENCHMARK
    }
    # path_merge = 'hirid_benchmark/merged_stage/'
    # patient_data_merge_stage = dd.read_parquet(path_merge)
    # patient_data_merge_stage = patient_data_merge_stage[
    #     ['patientid', 'datetime'] + [item[1] for item in metaids_physio.items()]]
    # patient_data_merge_stage = patient_data_merge_stage[patient_data_merge_stage['patientid'].isin(pid_valid)]
    # patient_data_merge_stage = patient_data_merge_stage.compute()
    # pickle.dump(patient_data_merge_stage, open(os.path.join(save_path, 'patient_data_merge_stage_selected.p'), 'wb'))
    # # patient_data_merge_stage = pickle.load(open(os.path.join(save_path, 'patient_data_merge_stage_selected.p'), 'rb'))
    #
    # print('Append APACHE and LOS to patient information table...')
    # # patient information with LOS and merged APACHE
    # patient_info = general_ext[general_ext['patientid'].isin(pid_valid)]
    # patient_info['APACHE MERGED'] = None
    # patient_info['LOS'] = None
    #
    # for pid in tqdm(pid_valid):
    #     assert len(apache_dict_merged[pid]) == 1
    #     patient_info.loc[patient_info['patientid'] == pid, 'APACHE MERGED'] = apache_dict_merged[pid][0]
    #     los = patient_data_merge_stage[patient_data_merge_stage['patientid'] == pid].shape[0] / 12 / 24
    #     patient_info.loc[patient_info['patientid'] == pid, 'LOS'] = los
    # pickle.dump(patient_info, open(os.path.join(save_path, 'patient_info.p'), 'wb'))
    # # patient_info = pickle.load(open(os.path.join(save_path, 'patient_info.p'), 'rb'))
    #
    # # filter patient with an ICU stay longer than 1 day
    # pid_los = patient_info.loc[(patient_info['LOS']>=1), 'patientid'].tolist()
    # pid_valid = [pid for pid in pid_valid if pid in pid_los]
    # print(f"Remove patient with LOS less than 1 day... --- {len(pid_valid)}")
    #
    # pickle.dump(pid_valid, open(os.path.join(save_path, 'pid_valid.p'), 'wb'))
    # # pid_valid = pickle.load(open(os.path.join(save_path, 'pid_valid.p'), 'rb'))

    ############################################################
    # Remove patient with invalid raw pharma data
    ############################################################
    print('Process raw pharma data...')
    # remove invalid entries
    print('\tRemove patients with invalid entries...')
    # vid_selected = [vid for k in MED_BENCHMARK_DICT_VID for vid in MED_BENCHMARK_DICT_VID[k]]
    # pharma_data = pd.read_parquet(pharma_raw_path)
    # pharma_data = pharma_data[(pharma_data['patientid'].isin(pid_valid))
    #                           & (pharma_data['pharmaid'].isin(vid_selected))
    #                           & (pharma_data['recordstatus'].isin(PHARMA_VALID))].reset_index(drop=True)
    # pid_0inj = pharma_data.loc[pharma_data['recordstatus']==544, 'patientid'].unique().tolist()
    # pid_valid = [pid for pid in pid_valid if pid not in pid_0inj]
    # pharma_data = pharma_data[pharma_data['patientid'].isin(pid_valid)]
    # # remove infusion with 0 cumulative dose
    # print('\tRemove 0 dose infusion...')
    # iid_0inf = pharma_data.loc[(pharma_data['recordstatus']==PHARMA_INFUSION_END) & (pharma_data['cumulativedose']==0), 'infusionid']
    # pharma_data = pharma_data[~pharma_data['infusionid'].isin(iid_0inf)]
    # # remove infusion with no start or end status
    # print('\tRemove patients with incomplete infusion...')
    # inf_no_start, inf_no_end = [], []
    # for iid in tqdm(pharma_data.loc[pharma_data['recordstatus'].isin(PHARMA_INFUSION), 'infusionid'].unique()):
    #     if PHARMA_INFUSION_START not in pharma_data.loc[pharma_data['infusionid'] == iid, 'recordstatus'].tolist():
    #         inf_no_start += [iid]
    #     if PHARMA_INFUSION_END not in pharma_data.loc[pharma_data['infusionid'] == iid, 'recordstatus'].tolist():
    #         inf_no_end += [iid]
    # pid_inf_not_complete = pharma_data.loc[pharma_data['infusionid'].isin(inf_no_start+inf_no_end), 'patientid'].unique().tolist()
    # print('#patients with incomplete infusion record: ', len(pid_inf_not_complete))
    # pid_valid = list(set(pid_valid).difference(set(pid_inf_not_complete)))
    # pickle.dump(pid_valid, open(os.path.join(save_path, 'pid_valid.p'), 'wb'))
    # # pid_valid = pickle.load(open(os.path.join(save_path, 'pid_valid.p'), 'rb'))
    #
    # # remove duplicates
    # print('\tRemove duplicated entries...')
    # pharma_data = drop_duplicates_pharma(pharma_data)
    # pickle.dump(pharma_data, open(os.path.join(save_path, 'pharma_data.p'), 'wb'))
    # # pharma_data = pickle.load(open(os.path.join(save_path, 'pharma_data.p'), 'rb'))
    #
    # pid_valid_with_selected_pharma = [pid for pid in pid_valid if pid in pharma_data['patientid'].unique()]
    # pickle.dump(pid_valid_with_selected_pharma, open(os.path.join(save_path, 'pid_valid_with_selected_pharma.p'), 'wb'))
    pid_valid_with_selected_pharma = pickle.load(open(os.path.join(save_path, 'pid_valid_with_selected_pharma.p'), 'rb'))


    ############################################################
    # Generate formatted pharma data per patient
    ############################################################
    print('Process grouped pharma data per patient...')
    # save_path_pharma_per_pat = os.path.join(save_path, 'pharma_per_pat')
    # Path(save_path_pharma_per_pat).mkdir(parents=True, exist_ok=True)
    #
    # with Pool(30) as pool:
    #     for _ in tqdm(
    #             pool.imap_unordered(
    #                 process_and_save_pharma_data_per_patient,
    #                 [dict(
    #                     pid=pid,
    #                     pharma_pat=pharma_data[pharma_data['patientid']==pid],
    #                     pharmaref=pharmaref,
    #                     save_path=save_path_pharma_per_pat,
    #                 ) for pid in pid_valid_with_selected_pharma] #[475:476]
    #             ), total=len(pid_valid_with_selected_pharma)
    #     ):
    #         pass


    ############################################################
    # Merge all data per patient
    ############################################################

    save_path_merged_data_per_pat = os.path.join(save_path, 'merged_data_per_pat')
    Path(save_path_merged_data_per_pat).mkdir(parents=True, exist_ok=True)

    path_endpoints = 'hirid_benchmark/endpoints'
    df_endpoints = pd.read_parquet(path_endpoints)
    #
    # print('Merge all data sources per patient...')
    # with Pool(30) as pool:
    #     for _ in tqdm(
    #             pool.imap_unordered(
    #                 merge_and_save_data_per_pat,
    #                 [dict(
    #                     save_path=save_path_merged_data_per_pat,
    #                     save_path_pharma_per_pat=save_path_pharma_per_pat,
    #                     pid=pid,
    #                     # df_data=patient_data_merge_stage[patient_data_merge_stage['patientid']==pid],
    #                     # df_endpoint=df_endpoints[df_endpoints['patientid']==pid],
    #                     # df_data=patient_data_merge_stage,
    #                     # df_endpoint=df_endpoints,
    #                     metaids_physio=metaids_physio,
    #                 ) for pid in pid_valid_with_selected_pharma]
    #             ), total=len(pid_valid_with_selected_pharma)
    #     ):
    #         pass

    ############################################################
    # Generate sample index -- medication
    ############################################################
    print('Generating training sample segments with medication...')
    THRES_MIN = 6
    THRES_BEFORE_MED = 3
    THRES_AFTER_MED = 3
    # sample_dict = {med: {} for med in MED_BENCHMARK}
    #
    # idx_sample = {med: 0 for med in MED_BENCHMARK}
    pid_sample = [int(pid.split('.')[0]) for pid in os.listdir(save_path_merged_data_per_pat)]
    # for pid in tqdm(pid_sample):
    #     df = pickle.load(open(os.path.join(save_path_merged_data_per_pat, f"{pid}.p"), 'rb'))
    #     for med in MED_BENCHMARK:
    #         med_data = df[med]
    #         med_data = med_data[med_data > 0]
    #         if med_data.shape[0] > 0:
    #             med_start = med_data.index[0]
    #             if med_start - df.index[0] >= timedelta(hours=THRES_MIN)\
    #                 and df.index[-1] - med_start >= timedelta(hours=THRES_AFTER_MED):
    #                 sample_start = med_start - timedelta(hours=THRES_BEFORE_MED)
    #                 sample_end = med_start + timedelta(hours=THRES_AFTER_MED)
    #                 sample_dict[med][idx_sample[med]] = (pid, sample_start, sample_end)
    #                 idx_sample[med] += 1
    # pickle.dump(sample_dict, open(os.path.join(save_path, 'sample_lookuptable_per_med.p'), 'wb'))
    sample_dict = pickle_load(os.path.join(save_path, 'sample_lookuptable_per_med.p'))

    for med in sample_dict:
        print(f"{med:<10} --- {len(sample_dict[med].keys())}")


    ############################################################
    # Generate training sample index -- vasopressors
    ############################################################
    # sample_dict = pickle.load(open(os.path.join(save_path, 'sample_lookuptable_per_med.p'), 'rb'))
    # MED_VASOPRESSOR = ['norepinephrine', 'epinephrine', 'dobutamine']
    # sample_dict_vasopressor = {}
    # sample_id = 0
    # print('Generating training sample segments with vasoactive agents...')
    # for med in MED_VASOPRESSOR:
    #     for i in sample_dict[med]:
    #         sample = sample_dict[med][i]
    #         sample_dict_vasopressor[sample_id] = [med, sample[0], sample[1], sample[2]]
    #         sample_id += 1
    # pickle.dump(sample_dict_vasopressor, open(os.path.join(save_path, 'sample_dict_vasopressor.p'), 'wb'))
    sample_dict_vasopressor = pickle_load(os.path.join(save_path, 'sample_dict_vasopressor.p'))


    ############################################################
    # Generate sample index -- control group with no vasoactive agents
    ############################################################
    print('Generating training sample segments of control group...')
    # sample_dict_control = {}
    # MED_VASO = MED_BENCHMARK[:3]
    # sample_id = 0
    # for pid in tqdm(pid_sample):
    #     df = pickle.load(open(os.path.join(save_path_merged_data_per_pat, f"{pid}.p"), 'rb'))
    #     df[MED_BENCHMARK] = df[MED_BENCHMARK].fillna(0)
    #
    #     t_start = df.index[0]
    #     t_end = t_start + timedelta(hours=THRES_MIN)
    #     t_last = df.index[-1]
    #     while t_last - t_end >= timedelta(hours=THRES_MIN):
    #         check = df.loc[(df.index>=t_start) & (df.index<t_end), MED_VASO].sum(axis=1)
    #         if check.sum() == 0:
    #             sample_dict_control[sample_id] = (pid, t_start, t_end)
    #             sample_id += 1
    #             t_start = t_end
    #         else:
    #             t_med_last = check.index[check>0][-1]
    #             if t_med_last == check.index[0]:
    #                 t_start = check.index[1]
    #             else:
    #                 t_start = t_med_last
    #         t_end = t_start + timedelta(hours=THRES_MIN)
    #
    # pickle.dump(sample_dict_control, open(os.path.join(save_path, 'sample_lookuptable_control_group.p'), 'wb'))
    sample_dict_control = pickle_load(os.path.join(save_path, 'sample_lookuptable_control_group.p'))

    ############################################################
    # Calculate normalization parameters for vasopressor samples
    ############################################################

    # split train-/test-dataset
    patient_info = pickle.load(open('processed-benchmark/patient_info.p', 'rb'))

    # # # stratify by patient discharge status
    # # dischargestatus_vaso = []
    # # for i in tqdm(sample_dict_vasopressor):
    # #     pid = sample_dict_vasopressor[i][1]
    # #     mortality = patient_info[patient_info['patientid'] == pid]['discharge_status'].item()
    # #     if mortality == 'alive':
    # #         dischargestatus_vaso.append(0)
    # #     elif mortality == 'dead':
    # #         dischargestatus_vaso.append(1)
    # #     else:
    # #         print(f"UNKNOWN STATUS: {mortality}")
    # #
    # # sample_train, sample_test = train_test_split(list(sample_dict_vasopressor.keys()),
    # #                                              test_size=0.2,
    # #                                              random_state=RANDOMSEED,
    # #                                              stratify=dischargestatus_vaso)


    # # stratify by medication type
    # med_label = []
    # for i in tqdm(sample_dict_vasopressor):
    #     med = sample_dict_vasopressor[i][0]
    #     if med == MED_BENCHMARK[0]:
    #         med_label.append(0)
    #     elif med == MED_BENCHMARK[1]:
    #         med_label.append(1)
    #     elif med == MED_BENCHMARK[2]:
    #         med_label.append(2)
    #     else:
    #         print('Unknown vasopressor!')
    #
    # sample_train, sample_test = train_test_split(list(sample_dict_vasopressor.keys()),
    #                                              test_size=0.2,
    #                                              random_state=RANDOMSEED,
    #                                              stratify=med_label)
    # pickle.dump(
    #     {'train': sample_train, 'test': sample_test},
    #     open(os.path.join(save_path, 'train_test_split_vasopressor.p'), 'wb')
    # )
    #
    # # Calculate normalization parameters with the train-set
    # df_statistics = []
    # for i in tqdm(sample_train):
    #     # med, pid, ts_start, ts_end = sample_dict_vasopressor[i]
    #     # sample = pickle.load(open(os.path.join(save_path_training_samples, med, f"{pid}.p"), 'rb'))
    #     sample = load_sample(save_path, sample_dict_vasopressor[i])
    #     df_statistics.append(sample)
    # for i in tqdm(sample_dict_control):
    #     sample = load_sample(save_path, sample_dict_control[i])
    #     df_statistics.append(sample)
    #
    # df_statistics = pd.concat(df_statistics)
    # df_statistics[MED_BENCHMARK] = df_statistics[MED_BENCHMARK].replace(.0, np.nan)
    #
    # norm_params = df_statistics.describe(percentiles=[.001, .01, .05, .1, .25, .5, .75, .9, .95, .99, .999])
    # pickle.dump(norm_params, open(os.path.join(save_path, 'norm_params_vasopressor.p'), 'wb'))
    norm_params = pickle_load(os.path.join(save_path, 'norm_params_vasopressor.p'))


    # calculate normalization parameters for patient information
    patient_info = pickle_load(os.path.join(save_path, 'patient_info.p'))

    df_statistics_info = []
    pids = [sample_dict_vasopressor[i][1] for i in sample_dict_vasopressor] + [sample_dict_control[i][0] for i in sample_dict_control]
    pids = list(set(pids))
    for pid in tqdm(pids):
        df_statistics_info.append(patient_info[patient_info['patientid'] == pid])
    df_statistics_info = pd.concat(df_statistics_info)
    norm_params_info = df_statistics_info.describe(percentiles=[.001, .01, .05, .1, .25, .5, .75, .9, .95, .99, .999])
    pickle_dump(norm_params_info, os.path.join(save_path, 'norm_params_info.p'))



    ############################################################
    # Filter sample by data missingness %
    ############################################################

    # med_labels = np.array([sample_dict[i][0] for i in range(len(sample_dict))])
    selected_physio = ['HR', 'RR', 'SpO2', 'ABPd', 'ABPm', 'ABPs', 'ZVD']
    selected_med = ['norepinephrine', 'epinephrine', 'dobutamine']

    dataset = MergedDataset(
        base_path=save_path,
        sample_dict=sample_dict_vasopressor,
        df_info=patient_info,
        type='vaso',
        norm=True, smooth=False, interpolate=False,
        selected_physio=selected_physio, selected_med=selected_med
    )

    dataset_control = MergedDataset(
        base_path=save_path,
        sample_dict=sample_dict_control,
        df_info=patient_info,
        type='control',
        norm=True, smooth=False, interpolate=False,
        selected_physio=selected_physio, selected_med=selected_med
    )


    for perc in [50, 60, 70, 80, 90]:
        sample_dict_filtered = {}
        count_discard = 0
        idx_new = 0
        for idx, sample in enumerate(tqdm(dataset)):
            missing1 = np.isnan(sample['data'][:90, :]).sum() / 90 / 7
            missing2 = np.isnan(sample['data'][90:, :]).sum() / 90 / 7

            if missing1 <= 1 - perc/100 or missing2 <= 1 - perc/100:
                sample_dict_filtered[idx_new] = sample_dict_vasopressor[idx]
                idx_new += 1
            else:
                count_discard += 1
        print(f"Remained - {len(sample_dict_filtered)}\tDiscarded - {count_discard}")
        pickle.dump(sample_dict_filtered,
                    open(os.path.join(save_path, f'sample_dict_vasopressor_filtered{perc}.p'), 'wb'))

    for perc in [50, 60, 70, 80, 90]:
        sample_dict_filtered = {}
        count_discard = 0
        idx_new = 0
        for idx, sample in enumerate(tqdm(dataset_control)):
            missing1 = np.isnan(sample['data'][:90, :]).sum() / 90 / 7
            missing2 = np.isnan(sample['data'][90:, :]).sum() / 90 / 7

            if missing1 <= 1 - perc/100 or missing2 <= 1 - perc/100:
                sample_dict_filtered[idx_new] = sample_dict_control[idx]
                idx_new += 1
            else:
                count_discard += 1
        print(f"Remained - {len(sample_dict_filtered)}\tDiscarded - {count_discard}")
        pickle.dump(sample_dict_filtered,
                    open(os.path.join(save_path, f'sample_dict_control_filtered{perc}.p'), 'wb'))

