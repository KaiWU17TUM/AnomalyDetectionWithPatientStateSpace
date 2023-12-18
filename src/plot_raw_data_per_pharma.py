import os
import pickle
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import matplotlib.dates as mdates
import datetime

import warnings
warnings.filterwarnings("ignore")

processed_path = 'processed-v2'
# apache = 'Surgical Cardiovascular'
apache = 'Cardiovascular'
xformatter = mdates.DateFormatter('%m/%d %H:%M')


def get_processed_path(file, processed_path=processed_path):
    return os.path.join(processed_path, file)


def read_patient_data(apache, pid, norm=True, processed_path=processed_path):
    if norm:
        file_path = get_processed_path(
            os.path.join('data_per_patient_resample2min_normalized',
                         apache.replace(' ', ''),
                         f'{pid}.csv')
        )
    else:
        file_path = get_processed_path(
            os.path.join('data_per_patient_resample2min',
                         apache.replace(' ', ''),
                         f'{pid}.csv')
        )

    return pd.read_csv(file_path, header=[0, 1], index_col=[0])


def get_vitals_individual(pharmaid, toi, pid, vitals=None, norm=True, apache=apache):
    if isinstance(toi, int):
        toi = [toi, toi]
    elif isinstance(toi, list):
        if len(toi) != 2:
            raise ValueError('Only integer or list of length 2 is accepted for toi.')
    else:
        raise ValueError('Only integer or list of length 2 is accepted for toi.')

    data = read_patient_data(apache=apache, pid=pid, norm=norm)
    data.index = pd.to_datetime(data.index)
    data.index.name = 'timestamp'
    pharma = data.loc[:, ('pharma', str(pharmaid))]
    ts_pharma = pharma.iloc[pharma.to_numpy().nonzero()].index
    #     if len(ts_pharma) == 0:
    #         print(f'Patient {pid} - No record for Pharma {pharmaid}.')

    for ts in ts_pharma:
        ts_start = ts - datetime.timedelta(minutes=60)
        ts_end = ts + datetime.timedelta(minutes=60)
        if (data.index[0] > ts_start) or (data.index[-1] < ts_end):
            continue

        data_ = data[(data.index >= ts_start) & (data.index <= ts_end)]
        data_ = pd.concat([data_], keys=[ts], names=['ts_pharma'])
        data_ = pd.concat([data_], keys=[pid], names=['patientid'])

    if 'data_' not in locals():
        #         print(data.index[0], data.index[-1], ts_pharma)
        return None

    if vitals:
        vitals = [str(v) for v in vitals]
        col0 = data_.columns.get_level_values(0)
        col1 = data_.columns.get_level_values(1)
        cond = ((col0 == "physio_num") & (col1 in vitals)) | (col0 == 'physio_cat') | (col0 == "pharma")
        data_ = data_.loc[:, cond]
        return data_

    return data_


def get_vitals(pharmaid: int, toi, pharma_pid_dict, vitals=None, norm=True,
               apache=apache):
    """ get patient physiological data before and after the use of medication
    pharmaid [int] - variable id of the one medicine
    toi [int/list] - time of interest in the unit of minute
        1. if toi is a list of two numbers [a, b]
           we return the physiological data within a min before the medicine and b min after the medicine
        2. if toi is an integer a
           the time interval is [a, a]
    vitals [list] - uid of numeric physio data
    """
    data = pd.DataFrame()
    for pid in tqdm(pharma_pid_dict[pharmaid]):
        data_ = get_vitals_individual(pharmaid, toi, pid, vitals, norm, apache)
        if data_ is not None:
            data = pd.concat([data, data_])

    return data





def plot_patient_data(data: pd.DataFrame, pid: int, pharmaid: int, save_path=None):
    """
    data: index - timestamp
          column - different signal sources
    """
    df_physio = data.loc[:, ('physio_num', slice(None))]
    df_pharma = data.loc[:, ('pharma', slice(None))]

    pharma_name = selected_pharma.loc[selected_pharma['variableid'] == pharmaid, 'variablename'].iloc[0]
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # plot timestamp of all medicines used in this periord
    for _, ts in df_pharma[(df_pharma != 0).any(1)].index:
        #         ts_loc = df_pharma.index.get_loc(ts)
        #         print(ts)
        plt.axvline(x=ts, color='gray', linestyle='--')
    plt.axvline(x=df_pharma.index[len(df_pharma.index) // 2][1], color='r', linestyle='--')

    # plot physio data
    for _, uid in df_physio.columns:
        label = selected_physio.loc[selected_physio['uid'] == int(uid), 'variablename'].unique().item()
        values = df_physio[('physio_num', uid)]
        values.index = values.index.droplevel(0)

        if (~pd.isnull(values)).sum() > 0:
            if (~pd.isnull(values)).sum() == 1:
                ax.scatter(values.index, values.values, label=label);
            else:
                ax.plot(values, label=label);

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.xticks(rotation=30, ha='right')
    # ax.xaxis.set_major_locator(plt.MaxNLocator(9))
    ax.xaxis.set_major_formatter(xformatter)
    ax.set_title(f'Patient {pid} - Pharma {pharmaid}: {pharma_name}', fontsize=16)
    plt.tight_layout()

    pharmatime = values.index.get_level_values(0).unique()[0]
    pharmatime = datetime.datetime.strftime(pharmatime, '%d%m%y-%H%M')

    if save_path:
        plt.savefig(os.path.join(save_path, str(pid) + '-' + pharmatime + '.png'))
    # plt.close()
    return

if __name__=='__main__':
    print(f'APACHE group: {apache}')
    ################################################################################################
    # load data
    ################################################################################################
    selected_pharma_file = os.path.join(processed_path, 'selected_pharma.p')
    selected_physio_file = os.path.join(processed_path, 'selected_physio.csv')
    selected_pharma = pickle.load(open(selected_pharma_file, 'rb'))
    selected_physio = pd.read_csv(selected_physio_file)

    pid = pickle.load(open(os.path.join(processed_path, 'pid_with_selected_pharma.p'), 'rb'))
    pid_group = pickle.load(open(os.path.join(processed_path, 'pid_group_valid.p'), 'rb'))
    patient_info = pickle.load(open(os.path.join(processed_path, 'patient_info_valid.p'), 'rb'))

    # # count patient number in each APACHE group
    # count_apache_group = {}
    # for apache in pid_group:
    #     count_apache_group[apache] = len(pid_group[apache])
    # count_apache_group = {k: v for k, v in sorted(count_apache_group.items(), key=lambda item: item[1], reverse=True)}

    ################################################################################################
    # count pharma occurrence in the APACHE patient group
    ################################################################################################
    pid_apache = sorted(pid_group[apache])
    save_path = os.path.join(processed_path, f'pharma_count_per_apache/')
    if os.path.exists(os.path.join(save_path, f'pharma_occurence_{apache.replace(" ", "")}.csv')):
        df_pharma_count_normed = pd.read_csv(os.path.join(save_path, f'pharma_occurence_{apache.replace(" ", "")}.csv'),
                                             index_col=[0])
    else:
        df_pharma_count_normed = pd.DataFrame(columns=sorted(selected_pharma['variableid'].unique()), index=pid_apache)
        for pid in tqdm(pid_apache):
            data = read_patient_data(apache=apache, pid=pid, norm=False)
            pharma = data.loc[:, ('pharma', slice(None))]
            df_pharma_count_normed.loc[pid] = pharma.astype(bool).sum().values

        Path(save_path).mkdir(parents=True, exist_ok=True)
        df_pharma_count_normed.to_csv(os.path.join(save_path, f'pharma_occurence_{apache.replace(" ", "")}.csv'))


    # select pharma that are used on more than 1000 patients for Surgical Cardiovascular
    # select pharma that are used on more than 400 patients for Cardiovascular
    if apache == 'Surgical Cardiovascular':
        num_thres = 1000
    elif apache == 'Cardiovascular':
        num_thres = 400
    else:
        raise ValueError('Only accept Cardiovascular or Surgical Cardiovascular as APACHE group for now.')
    pharma_count = df_pharma_count_normed.sum()
    pharma_occur = pharma_count.sort_values(ascending=False)
    pharma_count_per_patient = df_pharma_count_normed.astype(bool).sum()
    pharma_occur_per_patient = pharma_count_per_patient[pharma_count_per_patient > num_thres].sort_values(ascending=False)
    for variableid in pharma_occur_per_patient.index:
        variablename = selected_pharma.loc[selected_pharma['variableid'] == int(variableid), ['variablename']]
        num_occr = pharma_occur.loc[variableid]
        num_patient = pharma_occur_per_patient.loc[variableid]
        print(
            f'{variableid:<10} - {variablename.iloc[0].item():<50} - Pat. Occur: {num_patient:<6} - Occur: {num_occr:<8}')

    # generate dataset with given time-of-interest
    pharma_pid_dict = {}
    for pharma in tqdm(df_pharma_count_normed.columns):
        pids = df_pharma_count_normed.index[df_pharma_count_normed[pharma].to_numpy().nonzero()].to_list()
        pharma_pid_dict[int(pharma)] = pids


    for i, pharmaid in enumerate(pharma_occur_per_patient.index):
        pharmaid = int(pharmaid)
        pharmaname = selected_pharma.loc[selected_pharma['variableid']==int(pharmaid), ['variablename']].iloc[0].item()
        toi = [60, 60]
        print(f'Generating data samples {i:>3}/{len(pharma_occur_per_patient.index)} - {pharmaid:<10} - {pharmaname}')

        if os.path.exists(os.path.join(processed_path,
                                       f'sample_per_pharma/{apache.replace(" ", "")}/pharma{pharmaid}_{toi[0]}_{toi[1]}_{apache.replace(" ", "")}_normed.p')):
            df = pickle.load(open(os.path.join(
                processed_path,
                f'sample_per_pharma/{apache.replace(" ", "")}/pharma{pharmaid}_{toi[0]}_{toi[1]}_{apache.replace(" ", "")}_normed.p'), 'rb'))
            df_physio = df.loc[:, ('physio_num', slice(None))]
            df_pharma = df.loc[:, ('pharma', slice(None))]
        else:
            df = get_vitals(pharmaid, toi, apache=apache, norm=True, pharma_pid_dict=pharma_pid_dict)
            df_physio = df.loc[:, ('physio_num', slice(None))]
            df_pharma = df.loc[:, ('pharma', slice(None))]

            Path(os.path.join(processed_path, f'sample_per_pharma/{apache.replace(" ", "")}')).mkdir(exist_ok=True, parents=True)
            df[df == -1] = np.nan
            pickle.dump(df,
                        open(os.path.join(
                            processed_path,
                            f'sample_per_pharma/{apache.replace(" ", "")}/pharma{pharmaid}_{toi[0]}_{toi[1]}_{apache.replace(" ", "")}_normed.p'),
                            'wb'))

        # select physiological data with less missing data rate
        num_patient = df.index.get_level_values(0).unique().shape[0]
        missing_data_per_patient = df_physio.groupby('patientid').count().astype(bool).sum().sort_values(ascending=False).droplevel(0)
        num_entries = df.shape[0]
        missing_data = df_physio.fillna(0).astype(bool).sum().sort_values(ascending=False).droplevel(0)

        uid_selected = []
        uid_selected += missing_data_per_patient[missing_data_per_patient / num_patient > 0.1].index.tolist() \
                        + missing_data[missing_data / num_entries > 0.01].index.tolist()
        uid_selected = list(set(uid_selected))
        uid_selected = sorted(uid_selected, key=lambda x: missing_data_per_patient.loc[x], reverse=True)

        # save plots
        save_path = f'plots/pharma_response/{apache.replace(" ", "")}/{pharmaid}/'
        Path(save_path).mkdir(parents=True, exist_ok=True)

        pid_pharma = df.index.get_level_values(0).unique()
        print(f'Saving plot.')
        for pid in tqdm(pid_pharma):
            data = df.loc[pid]
            plot_patient_data(data, pid, pharmaid, save_path);

