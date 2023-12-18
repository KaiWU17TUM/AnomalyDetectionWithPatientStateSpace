import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler
from scipy import stats

from utils.config_dataset import *


os.chdir('/home/kai/DigitalICU/Experiments/HIRID-PatientStateSpace/src')




def percentile(n):
    def percentile_(x):
        return np.nanpercentile(x, n)

    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def robust_normalize(x, norm_param, perc_lower, perc_upper, unit=False):
    pl = norm_param.loc[str(perc_lower)]
    pu = norm_param.loc[str(perc_upper)]
    c = 1
    if unit:
        c = stats.norm.ppf(perc_upper/100) - stats.norm.ppf(perc_lower/100)
    x_norm = (x - pl) / (pu - pl) / c

    return x_norm



def format_data(idx_ts, info, physio, pharma, val_cat, freq='2T', norm=False):
    cols_num = COL_PHYSIO_NUM
    cols_cat = [f'{uid}_{val}' for uid in COL_PHYSIO_CAT for val in val_cat[uid]]
    cols_pha = COL_PHARMA
    # cols = cols_num + cols_cat + cols_pha

    # df = pd.DataFrame(columns=cols, index=idx_ts)
    df_num = format_numeric_data(idx_ts, physio, cols_num, freq, norm)
    df_cat = format_categoric_data(idx_ts, physio, cols_cat, freq)
    df_pha = format_pharma_data(idx_ts, pharma, cols_pha, freq)

    df = pd.concat([df_num, df_cat, df_pha], axis=1, keys=['physio_num', 'physio_cat', 'pharma'])
    # df.update(df_num)
    # df.update(df_cat)
    # df.update(df_pha)
    return df



def format_numeric_data(idx_ts, data, cols_num, freq='2T', norm=False):
    # idx_ts: resample timestamp starting from admission time with a frequency of 2min
    # data: raw dataframe
    # col_num: column names in the output dataframe
    # df: formated dataframe of shape T x D
    df = pd.DataFrame(columns=cols_num, index=idx_ts)

    for col in set(COL_PHYSIO_NUM).difference(set(COL_PHYSIO_FLUID)):
        d = data[data['uid']==col].set_index('datetime')['value']
        if d.shape[0] == 0:
            continue
        d = d.resample(freq, origin=idx_ts[0]).mean()
        df_col = pd.DataFrame(index=d.index)
        df_col[col] = d.values
        df.update(df_col)

    for col in COL_PHYSIO_FLUID:
        d = data[data['uid'] == col].set_index('datetime')['value']
        if d.shape[0] == 0:
            continue
        # Urine [ml/h]
        if col == 43:
            d = d.resample(freq, origin=idx_ts[0]).mean()
        # drain [ml]
        elif col == 44:
            d = d.resample(freq, origin=idx_ts[0]).sum()
        # fluid balance [ml]: cumulative
        elif (col == 45) or (col == 46):
            d = d.resample(freq, origin=idx_ts[0]).last()
        else:
            raise ValueError(f'UID-{col} is not in patient fluid data')
        df_col = pd.DataFrame(index=d.index)
        df_col[col] = d.values
        df.update(df_col)

    if norm:
        df = df.fillna(-1)

    return df


def format_categoric_data(idx_ts, data, cols_cat, freq='2T'):
    df = pd.DataFrame(0, columns=cols_cat, index=idx_ts)

    # One hot encoding
    for col in COL_PHYSIO_CAT:
        d = data[data['uid']==col].set_index('datetime')
        if d.shape[0] == 0:
            continue
        d_ohk = pd.get_dummies(d['value'].astype(int), prefix=col).resample(freq, origin=idx_ts[0]).last()
        df.update(d_ohk)

    return df


def format_pharma_data(idx_ts, data, cols_pha, freq='2T'):
    df = pd.DataFrame(0, columns=cols_pha, index=idx_ts)

    for col in data['pharmaid'].unique():
        try:
            # if col == 1000974:
            #     pass
            d = data[(data['pharmaid'] == col) & (data['givendose']!=0)].set_index('givenat')['givendose']
            if d.shape[0] == 0:
                continue
            d = d.resample(freq, origin=idx_ts[0]).sum()
            df_col = pd.DataFrame(index=d.index)
            df_col[col] = d.values
            df.update(df_col)
        except:
            print(f'Patient {data["patientid"][0]} - Pharma {col}')

    return df

