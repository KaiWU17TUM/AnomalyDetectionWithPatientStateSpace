import os
import pickle
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


import warnings
warnings.filterwarnings("ignore")


from utils.config_dataset import *

import matplotlib.dates as mdates
import datetime

xformatter = mdates.DateFormatter('%m/%d %H:%M')


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
    plt.close()
    return