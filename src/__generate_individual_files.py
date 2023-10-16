import os
import pickle
import collections
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
from multiprocessing import Pool

import sys
sys.path.append(os.path.abspath('.'))
# print(sys.path)
from utils.preprocess import format_data, robust_normalize


def generate_individual_files(args):
    pid = args['pid']
    info = args['info']
    val_cat = args['val_cat']
    freq = args['freq']
    norm = args['norm']
    save_path = args['save_path']

    if norm:
        pharma = pd.read_csv(f'processed/data_per_patient_normalized/{pid}_pharma.csv')
        physio = pd.read_csv(f'processed/data_per_patient_normalized/{pid}_physio.csv')
        pharma['givenat'] = pd.to_datetime(pharma['givenat'])
        physio['datetime'] = pd.to_datetime(physio['datetime'])
        pharma = pharma.sort_values('givenat')
        physio = physio.sort_values('datetime')

        t_start = info['admissiontime'].item()
        t_end = max(physio['datetime'].max(), pharma['givenat'].max())
        idx_ts = pd.date_range(start=t_start, end=t_end, freq=freq)

        df = format_data(idx_ts, info, physio, pharma, val_cat, freq=freq, norm=norm)
        df.to_csv(os.path.join(save_path, f'{pid}.csv'))

    else:
        pharma = pd.read_csv(f'processed/data_per_patient/{pid}_pharma.csv')
        physio = pd.read_csv(f'processed/data_per_patient/{pid}_physio.csv')
        pharma['givenat'] = pd.to_datetime(pharma['givenat'])
        physio['datetime'] = pd.to_datetime(physio['datetime'])
        pharma = pharma.sort_values('givenat')
        physio = physio.sort_values('datetime')

        t_start = info['admissiontime'].item()
        t_end = max(physio['datetime'].max(), pharma['givenat'].max())
        idx_ts = pd.date_range(start=t_start, end=t_end, freq=freq)

        df = format_data(idx_ts, info, physio, pharma, val_cat, freq=freq, norm=norm)
        df.to_csv(os.path.join(save_path, f'{pid}.csv'))



def generate_data_sample_per_pharma(args):
    T_BUFFER = args['T_BUFFER']
    T_PREV = args['T_PREV']
    T_AFTER = args['T_AFTER']
    pid = args['pid']
    pharma_all = args['pharma_all']
    freq = args['freq']
    # pharma_all = pharma_all[pharma_all['patientid']==pid]

    df = pd.read_csv(f'processed/data_per_patient_resample2min_normalized/{pid}.csv', header=[0,1], index_col=0)
    df.index = pd.to_datetime(df.index)
    pharma = df.loc[:, df.columns.get_level_values(0)=='pharma']
    # ignore empty rows and columns
    pharma = pharma.loc[(pharma!=0).any(1), pharma.columns[(pharma!=0).any(0)]]

    # align timestamp for raw pharma data
    pharma_all = pharma_all[['pharmaid', 'givenat']]
    pharma_all['givenat'] = pd.to_datetime(pharma_all['givenat'])
    pharma_all = pharma_all.sort_values('givenat').set_index('givenat')
    pharma_all = pharma_all.resample(freq, origin=df.index[0]).last()
    pharma_all = pharma_all.loc[(~(pd.isnull(pharma_all))).any(1)]

    for pharmaid in pharma.columns.get_level_values(1).to_numpy().astype(int):
        Path(f'processed/data_sample_per_pharma_1h/{pharmaid}').mkdir(parents=True, exist_ok=True)

        this_pharma = pharma[pharma.columns[pharma.columns.get_level_values(1) == str(pharmaid)]]
        this_pharma = this_pharma[(this_pharma!=0).any(1)]
        this_pharma.index = pd.to_datetime(this_pharma.index)
        ts_idx = 0
        ts_curr = pd.to_datetime(this_pharma.index[ts_idx])

        sampleid = 1

        T_BEGIN = pd.to_datetime(df.index[0])
        T_END = pd.to_datetime(df.index[-1])
        while ts_curr + datetime.timedelta(hours=T_AFTER) <= T_END:
            # check if there is enough data for prev-med data
            if ts_curr - T_BEGIN < datetime.timedelta(hours=T_PREV):
                ts_idx += 1
                try:
                    ts_curr = pd.to_datetime(this_pharma.index[ts_idx])
                except:
                    break
                continue

            # ts_last_med = pharma_all.loc[(pharma_data_all['pharmaid']!=pharmaid)
            #                              & (pharma_data_all['givenat']<this_pharma.index[0])].max()['givenat']
            ts_last_med = pharma_all.loc[pharma_all.index < ts_curr]
            if ts_last_med.shape[0] == 0:
                dt_prev_h = None
            else:
                ts_last_med = ts_last_med.index.max()
                dt_prev = pd.to_datetime(ts_curr) - ts_last_med
                dt_prev_h = dt_prev.days * 24 + dt_prev.seconds / 3600


            ts_next_med = pharma_all.loc[(pharma_all['pharmaid']!=pharmaid) & (pharma_all.index>ts_curr)]
            if ts_next_med.shape[0] == 0:
                dt_next_h = None
            else:
                ts_next_med = ts_next_med.index.min()
                dt_next = ts_next_med - pd.to_datetime(ts_curr)
                dt_next_h = dt_next.days * 24 + dt_next.seconds / 3600


            ts_begin = ts_curr - datetime.timedelta(hours=T_PREV)
            ts_end = ts_curr + datetime.timedelta(hours=T_AFTER)

            if dt_prev_h: # check time interval to previous medicine
                try:
                    if datetime.timedelta(hours=dt_prev_h) < datetime.timedelta(hours=T_BUFFER + T_PREV):
                        ts_idx += 1
                        try:
                            ts_curr = pd.to_datetime(this_pharma.index[ts_idx])
                        except:
                            break
                        continue
                    else:
                        if dt_next_h: # check time interval to next medicine
                            if datetime.timedelta(hours=dt_next_h) < datetime.timedelta(hours=T_AFTER):
                                ts_idx += 1
                                try:
                                    ts_curr = pd.to_datetime(this_pharma.index[ts_idx])
                                except:
                                    break
                                continue
                    sample = df.loc[(df.index >= ts_begin) & (df.index <= ts_end)]
                    print(f'\tSaving valid data sample --- {pharmaid} / {pid}_{sampleid} - {sample.index[0]} - {ts_curr}')
                    sample.to_csv(f'processed/data_sample_per_pharma_1h/{pharmaid}/{pid}_{sampleid}.csv')
                    sampleid += 1
                    next_idx = this_pharma[this_pharma.index > pd.to_datetime(sample.index[-1])]
                    if next_idx.shape[0] == 0:
                        break
                    ts_idx = this_pharma.index.to_list().index(next_idx.index.min())
                    ts_curr = pd.to_datetime(this_pharma.index[ts_idx])
                except:
                    print(dt_prev_h)

            else:
                try:
                    if dt_next_h:
                        if datetime.timedelta(hours=dt_next_h) < datetime.timedelta(hours=T_AFTER):
                            ts_idx += 1
                            try:
                                ts_curr = pd.to_datetime(this_pharma.index[ts_idx])
                            except:
                                break
                            continue
                    sample = df.loc[(df.index >= ts_begin) & (df.index <= ts_end)]
                    print(f'\tSaving valid data sample --- {pharmaid} / {pid}_{sampleid} - {sample.index[0]} - {ts_curr}')
                    sample.to_csv(f'processed/data_sample_per_pharma_1h/{pharmaid}/{pid}_{sampleid}.csv')
                    sampleid += 1
                    next_idx = this_pharma[this_pharma.index > pd.to_datetime(sample.index[-1])]
                    if next_idx.shape[0] == 0:
                        break
                    ts_idx = this_pharma.index.to_list().index(next_idx.index.min())
                    ts_curr = pd.to_datetime(this_pharma.index[ts_idx])
                except:
                    print(dt_next_h)



if __name__=='__main__':

    # selected_pharma = pd.read_csv('processed/HiRID_selected_variables-input_dict.csv')
    selected_pharma = pickle.load(open('processed/selected_pharma_final.p', 'rb'))
    selected_physio = pd.read_csv('processed/HiRID_selected_variables-output_processed.csv')

    pid_list = pickle.load(open('processed/pid_valid.p', 'rb'))
    pid_group = pickle.load(open('processed/pid_group_valid.p', 'rb'))
    patient_info = pickle.load(open('processed/patient_info_valid.p', 'rb'))
    # pharma_data = pickle.load(open('processed/pharma_data_valid.p', 'rb'))
    # patient_data = pickle.load(open('processed/patient_data_valid_with_uid.p', 'rb'))

    pharma_data_all = pd.read_parquet(os.path.join('physionet.org/files/hirid/1.1.1/raw_stage/pharma_records_parquet',
                                                   'pharma_records', 'parquet'))

    # selected_pharma = selected_pharma.drop(
    #     selected_pharma[~selected_pharma['variableid'].isin(pharma_data['pharmaid'].unique())].index
    # ).reset_index(drop=True)
    # pickle.dump(selected_pharma, open('processed/selected_pharma_final.p', 'wb'))

    COL_PHARMA = selected_pharma['variableid'].tolist()
    COL_PHYSIO_NUM = selected_physio.loc[selected_physio['type'] == 'n', 'uid'].unique().tolist()
    COL_PHYSIO_CAT = selected_physio.loc[selected_physio['type'] == 'c', 'uid'].unique().tolist()
    COL_PHYSIO_FLUID = selected_physio.loc[selected_physio['category'] == 'Fluid-balance', 'uid'].unique().tolist()

    # normalize numeric data
    perc_lower = 0.5
    perc_upper = 99.5
    norm_param_physio = pd.read_csv('processed/robnorm_pararms_physio.csv', index_col=[0])
    norm_param_pharma = pd.read_csv('processed/robnorm_pararms_pharma.csv', index_col=[0])
    # patient_data_normed = patient_data.copy(deep=True)
    # pharma_data_normed = pharma_data.copy(deep=True)
    # for uid in tqdm(COL_PHYSIO_NUM):
    #     try:
    #         norm_param = norm_param_physio.loc[uid]
    #         pl = norm_param.loc[str(perc_lower)]
    #         pu = norm_param.loc[str(perc_upper)]
    #         len_prev = patient_data_normed.shape[0]
    #         patient_data_normed = patient_data_normed.loc[
    #             ((patient_data_normed['uid']==uid) & (patient_data_normed['value']>=pl) & (patient_data_normed['value']<=pu))
    #             | (patient_data_normed['uid']!=uid)
    #             ]
    #         # check1 = patient_data_normed[(patient_data_normed['uid']==uid)]
    #         # check2 = patient_data_normed[(patient_data_normed['uid'] == uid) &
    #         #                              (patient_data_normed['value'] >= pl) & (patient_data_normed['value'] <= pu)]
    #         # check3 = patient_data_normed[(patient_data_normed['uid'] == uid) &
    #         #                              ((patient_data_normed['value'] < pl) | (patient_data_normed['value'] > pu))]
    #         print(f"Removed data: {patient_data.shape[0]} - {len_prev} - {patient_data_normed.shape[0]} : {patient_data_normed.shape[0]-len_prev}")
    #
    #         data = patient_data_normed.loc[patient_data_normed['uid']==uid, 'value']
    #         data_normed = robust_normalize(data, norm_param, perc_lower, perc_upper)
    #         patient_data_normed.loc[patient_data_normed['uid'] == uid, 'value'] = data_normed
    #     except:
    #         raise ValueError('check norm_params_physio!')
    # pickle.dump(patient_data_normed, open('processed/patient_data_valid_normalized.p', 'wb'))
    #
    # for vid in tqdm(COL_PHARMA):
    #     try:
    #         norm_param = norm_param_pharma.loc[vid]
    #         pl = norm_param.loc[perc_lower]
    #         pu = norm_param.loc[perc_upper]
    #         len_prev = pharma_data_normed.shape[0]
    #         pharma_data_normed = pharma_data_normed.loc[
    #             ((pharma_data_normed['pharmaid']==vid) & (pharma_data_normed['givendose'] > pl) & (pharma_data_normed['givendose'] < pu))
    #             | (pharma_data_normed['pharmaid']!=vid)
    #             ]
    #         print(f"Removed data: {pharma_data.shape[0]} - {len_prev} - {pharma_data_normed.shape[0]} : {pharma_data_normed.shape[0] - len_prev}")
    #
    #         data = pharma_data_normed.loc[pharma_data_normed['pharmaid'] == vid, 'givendose']
    #         data_normed = robust_normalize(data, norm_param, perc_lower, perc_upper)
    #         pharma_data_normed.loc[pharma_data_normed['pharmaid'] == vid, 'givendose'] = data_normed
    #     except:
    #         pass
    # pickle.dump(pharma_data_normed, open('processed/pharma_data_valid_normalized.p', 'wb'))

    # patient_data_normalized = pickle.load(open('processed/patient_data_valid_normalized.p', 'rb'))
    # pharma_data_normalized = pickle.load(open('processed/pharma_data_valid_normalized.p', 'rb'))

    # check_pharma = pharma_data_normalized.groupby('pharmaid').agg(
    #     {'givendose': ['median', 'mean', 'std', 'min', 'max']}
    # )
    # check_physio = patient_data_normalized.groupby('uid').agg(
    #     {'value': ['median', 'mean', 'std', 'min', 'max']}
    # )
    #
    # # SAVE RAW DATA IN INDIVIDUAL FILES FOR FURTHER PREPROCESSING IN PARALLEL
    # Path('processed/data_per_patient').mkdir(parents=True, exist_ok=True)
    # Path('processed/data_per_patient_normalized').mkdir(parents=True, exist_ok=True)
    # for pid in tqdm(pid_list):
    #     pharma = pharma_data[pharma_data['patientid']==pid].reset_index(drop=True)
    #     physio = patient_data[patient_data['patientid']==pid].reset_index(drop=True)
    #     pharma.to_csv(f'processed/data_per_patient/{pid}_pharma.csv')
    #     physio.to_csv(f'processed/data_per_patient/{pid}_physio.csv')
    #     physio_normed = patient_data_normalized[patient_data_normalized['patientid']==pid].reset_index(drop=True)
    #     pharma_normed = pharma_data_normalized[pharma_data_normalized['patientid'] == pid].reset_index(drop=True)
    #     pharma_normed.to_csv(f'processed/data_per_patient_normalized/{pid}_pharma.csv')
    #     physio_normed.to_csv(f'processed/data_per_patient_normalized/{pid}_physio.csv')


    # # Format raw data into shape TxD
    # Path('processed/data_per_patient_resample2min').mkdir(parents=True, exist_ok=True)
    # Path('processed/data_per_patient_resample2min_normalized').mkdir(parents=True, exist_ok=True)
    #
    # val_cat = {}
    # for uid in COL_PHYSIO_CAT:
    #     val_cat[uid] = sorted(patient_data[patient_data['uid'] == uid]['value'].astype(int).unique())
    #
    # with Pool(48) as pool:
    #     for _ in tqdm(
    #             pool.imap_unordered(
    #                 generate_individual_files,
    #                 [dict(
    #                     pid=pid,
    #                     info=patient_info[patient_info['patientid'] == pid],
    #                     val_cat=val_cat,
    #                     freq='2T',
    #                     norm=True,
    #                     save_path='processed/data_per_patient_resample2min_normalized',
    #                 ) for pid in pid_list]
    #             ), total=len(pid_list)
    #     ):
    #         pass





    # Generate data samples: N hours after taking a medicine with no other medecine taken simultaneously
    T_BUFFER = 1
    T_PREV = 1
    T_AFTER = 1

    with Pool(48) as pool:
        for _ in tqdm(
                pool.imap_unordered(
                    generate_data_sample_per_pharma,
                    [dict(
                        T_BUFFER=T_BUFFER,
                        T_PREV=T_PREV,
                        T_AFTER=T_AFTER,
                        pid=pid,
                        pharma_all=pharma_data_all[pharma_data_all['patientid']==pid],
                        freq='2T',
                    ) for pid in [pid_list]]
                ), total=len(pid_list)
        ):
            pass



    # T_BUFFER = 1
    # T_PREV = 1
    # T_AFTER = 1
    # for pid in tqdm(pid_list):
    #     df = pd.read_csv(f'processed/data_per_patient_resample2min_normalized/{pid}.csv', header=[0,1], index_col=0)
    #     pharma_all = pharma_data_all[pharma_data_all['patientid'] == pid]
    #
    #     pharma = df.loc[:, df.columns.get_level_values(0)=='pharma']
    #     # ignore empty rows and columns
    #     pharma = pharma.loc[(pharma!=0).any(1), pharma.columns[(pharma!=0).any(0)]]
    #
    #     for pharmaid in pharma.columns.get_level_values(1).to_numpy().astype(int):
    #         Path(f'processed/data_sample_per_pharma_1h/{pharmaid}').mkdir(parents=True, exist_ok=True)
    #
    #         this_pharma = pharma[pharma.columns[pharma.columns.get_level_values(1) == str(pharmaid)]]
    #         this_pharma = this_pharma[(this_pharma!=0).any(1)]
    #         ts_idx = 0
    #         ts_curr = pd.to_datetime(this_pharma.index[ts_idx])
    #
    #         sampleid = 1
    #
    #         T_BEGIN = pd.to_datetime(df.index[0])
    #         T_END = pd.to_datetime(df.index[-1])
    #         while ts_curr + datetime.timedelta(hours=T_AFTER) <= T_END:
    #             # check if there is enough data for prev-med data
    #             if ts_curr - T_BEGIN < datetime.timedelta(hours=T_PREV):
    #                 ts_idx += 1
    #                 try:
    #                     ts_curr = pd.to_datetime(this_pharma.index[ts_idx])
    #                 except:
    #                     break
    #                 continue
    #             try:
    #                 # ts_last_med = pharma_all.loc[(pharma_data_all['pharmaid']!=pharmaid)
    #                 #                              & (pharma_data_all['givenat']<this_pharma.index[0])].max()['givenat']
    #                 ts_last_med = pharma_all.loc[pharma_data_all['givenat'] < this_pharma.index[0]].max()['givenat']
    #                 dt_prev = pd.to_datetime(this_pharma.index[0]) - ts_last_med
    #                 dt_prev_h = dt_prev.days * 24 + dt_prev.seconds / 3600
    #             except:
    #                 dt_prev_h = None
    #             try:
    #                 ts_next_med = pharma_all.loc[(pharma_data_all['pharmaid']!=pharmaid)
    #                                              & (pharma_data_all['givenat']>this_pharma.index[-1])].min()['givenat']
    #                 dt_next = ts_next_med - pd.to_datetime(this_pharma.index[-1])
    #                 dt_next_h = dt_next.days * 24 + dt_next.seconds / 3600
    #             except:
    #                 dt_next_h = None
    #
    #             if dt_prev_h: # check time interval to previous medicine
    #                 try:
    #                     if datetime.timedelta(hours=dt_prev_h) < datetime.timedelta(hours=T_BUFFER + T_PREV):
    #                         ts_idx += 1
    #                         try:
    #                             ts_curr = pd.to_datetime(this_pharma.index[ts_idx])
    #                         except:
    #                             break
    #                         continue
    #                     else:
    #                         if dt_next_h: # check time interval to next medicine
    #                             if datetime.timedelta(hours=dt_next_h) < datetime.timedelta(hours=T_AFTER):
    #                                 ts_idx += 1
    #                                 try:
    #                                     ts_curr = pd.to_datetime(this_pharma.index[ts_idx])
    #                                 except:
    #                                     break
    #                                 continue
    #                     ts_begin = ts_curr - datetime.timedelta(hours=T_BUFFER + T_PREV)
    #                     ts_end = ts_curr + datetime.timedelta(hours=T_AFTER)
    #                     sample = df.loc[(pd.to_datetime(df.index) >= ts_begin) & (pd.to_datetime(df.index) <= ts_end)]
    #                     sample.to_csv(f'processed/data_sample_per_pharma_1h/{pharmaid}/{pid}_{sampleid}.csv')
    #                     sampleid += 1
    #                     ts_idx = np.argmax(pd.to_datetime(this_pharma.index) > pd.to_datetime(sample.index[-1]))
    #                     ts_curr = pd.to_datetime(this_pharma.index[ts_idx])
    #                 except:
    #                     pass
    #
    #             else:
    #                 if dt_next_h:
    #                     if pd.to_datetime(dt_next_h) < datetime.timedelta(hours=T_AFTER):
    #                         ts_idx += 1
    #                         try:
    #                             ts_curr = pd.to_datetime(this_pharma.index[ts_idx])
    #                         except:
    #                             break
    #                         continue
    #                 ts_begin = ts_curr - datetime.timedelta(hours=T_PREV)
    #                 ts_end = ts_curr + datetime.timedelta(hours=T_AFTER)
    #                 sample = df.loc[(pd.to_datetime(df.index) >= ts_begin) & (pd.to_datetime(df.index) <= ts_end)]
    #                 sample.to_csv(f'processed/data_sample_per_pharma_1h/{pharmaid}/{pid}_{sampleid}.csv')
    #                 sampleid += 1
    #                 ts_idx = np.argmax(pd.to_datetime(this_pharma.index) > pd.to_datetime(sample.index[-1]))
    #                 ts_curr = pd.to_datetime(this_pharma.index[ts_idx])




    # Generate data samples: N hours after taking medicines allowing taking other medicines inbetween