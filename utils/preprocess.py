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

    # df = pd.concat([df_num, df_cat, df_pha], axis=1, keys=['physio_num', 'physio_cat', 'pharma'])
    df = pd.concat([df_num, df_cat, df_pha], axis=1, keys=['physio_num', 'physio_cat'])
    df = pd.concat([df, df_pha], axis=1)

    return df



def format_numeric_data(idx_ts, data, cols_num, freq='2T', norm=False):
    # idx_ts: resample timestamp starting from admission time with a frequency of 2min
    # data: raw dataframe
    # col_num: column names in the output dataframe
    # df: formated dataframe of shape T x D
    data = data[(data['datetime'] >= idx_ts[0]) & (data['datetime'] <= idx_ts[-1])]
    df = pd.DataFrame(columns=cols_num, index=idx_ts)

    for col in set(COL_PHYSIO_NUM).difference(set(COL_PHYSIO_FLUID)):
        d = data[data['uid']==col].copy().set_index('datetime')['value']
        if d.shape[0] == 0:
            continue
        d = d.resample(freq, origin=idx_ts[0]).mean()
        df_col = pd.DataFrame(index=d.index)
        df_col[col] = d.values
        df.update(df_col)

    for col in COL_PHYSIO_FLUID:
        # d = data[data['uid'] == col].set_index('datetime')['value']
        d = data[data['uid'] == col].copy()[['datetime', 'value']]
        if d.shape[0] == 0:
            continue
        # Urine [ml/h]
        if col == 43:
            d = d.set_index('datetime')
            d = d.resample(freq, origin=idx_ts[0]).mean()
        # drain [ml]
        elif col == 44:
            d = d.set_index('datetime')
            d = d.resample(freq, origin=idx_ts[0]).sum()
        # fluid balance [ml]: cumulative
        elif (col == 45) or (col == 46):
            # d['diff'] = d['value'].diff()
            # d['diff_t'] = d['datetime'].diff() / np.timedelta64(1,"m")
            d['rate'] = np.nan
            d['rate'][:-1] = d['value'].diff().values[1:] / (d['datetime'].diff() / np.timedelta64(1,"m")).values[1:]
            del d['value']
            d = d.set_index('datetime')
            d = d.resample(freq, origin=idx_ts[0]).last()
        else:
            raise ValueError(f'UID-{col} is not in patient fluid data')
        df_col = pd.DataFrame(index=df.index, columns=[col])
        df_col.loc[d.index, col] = d.values.reshape(-1)
        if (col == 43):
            df_col = df_col.fillna(method='bfill')
        elif  (col == 45) or (col == 46):
            df_col[df_col.index <= d.index[-1]] = df_col[df_col.index <= d.index[-1]].fillna(method='ffill')

        # df_col[col] = d.values
        df.update(df_col)

    if norm:
        df = df.fillna(-1)

    return df


def format_categoric_data(idx_ts, data, cols_cat, freq='2T'):
    data = data[(data['datetime'] >= idx_ts[0]) & (data['datetime'] <= idx_ts[-1])]
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
    df = pd.DataFrame(0,
                      columns=[('pharma', pha) for pha in cols_pha]
                              + [('pharma_mask', pha) for pha in cols_pha],
                      index=idx_ts)
    data = data[data['recordstatus'].isin(PHARMA_VALID)]

    for pharmaid in data['pharmaid'].unique():
        df_col = pd.DataFrame(index=df.index, columns=[('pharma', pharmaid), ('pharma_mask', pharmaid)])
        df_col[('pharma', pharmaid)] = np.nan
        df_col[('pharma_mask', pharmaid)] = np.nan

        data_ = data[data['pharmaid']==pharmaid]
        for iid in data_['infusionid'].unique():
            d = data_[data_['infusionid'] == iid].copy().sort_values('givenat')
            d['mask'] = 1
            d['rate'] = 0
            if set(d['recordstatus']).issubset(PHARMA_INFUSION):
                check_fisrt_status = d.iloc[0].recordstatus
                try:
                    assert((d['recordstatus']==PHARMA_INFUSION_START).sum() == 1)
                except AssertionError:
                    acting_period = pharmaref[pharmaref['pharmaid']==pharmaid].iloc[0].pharmaactingperiod_min
                    beg_time = d.iloc[0]['givenat'] - np.timedelta64(acting_period, "m")
                    d.set_index('givenat', inplace=True)
                    d.loc[beg_time, 'givendose'] = 0
                    d.loc[beg_time, 'recordstatus'] = PHARMA_INFUSION_START
                    d.loc[beg_time, 'infusionid'] = iid
                    d.sort_index(inplace=True)
                    d.reset_index(inplace=True)
                try:
                    assert ((d['recordstatus'] == PHARMA_INFUSION_END).sum() == 1)
                except AssertionError:
                    pass
                d['rate'][:-1] = d['givendose'].values[1:] / (d['givenat'].diff() / np.timedelta64(1, "m")).values[1:]
                d = d[['givenat', 'rate', 'mask']].set_index('givenat')
                d = d.resample(freq, origin=idx_ts[0]).last()
                d = d[(d.index >= idx_ts[0]) & (d.index <= idx_ts[-1])]

                df_col.loc[d.index, [('pharma', pharmaid)]] = d['rate']
                df_col.loc[d.index, [('pharma_mask', pharmaid)]] = d['mask']
                # df_col.loc[d.index,:] = df_col.loc[d.index,:].replace(to_replace=0, method='ffill').values
                df_col.loc[d.index, :] = df_col.loc[d.index, :].fillna(method='ffill').values

            elif set(d['recordstatus']).issubset(PHARMA_INJECTION):
                if d.shape[0] != 1:
                    # print(f"More than 1 entry for Injection of InfusionID {iid} -- {d['givenat'].values}.")
                    pass
                d['rate'] = d['givendose']
                d = d[['givenat', 'rate', 'mask']].set_index('givenat')
                d = d.resample(freq, origin=idx_ts[0]).sum()
                d = d[(d.index >= idx_ts[0]) & (d.index <= idx_ts[-1])]

                df_col.loc[d.index, [('pharma', pharmaid)]] = d['rate']
                df_col.loc[d.index, [('pharma_mask', pharmaid)]] = d['mask']

            else:
                raise ValueError(f'InfusionID {iid} is not a valid infusion or injection -- {d["recordstatus"].unique()}.')

        # df.update(df_col)
        df_col = df_col[(df_col.index >= idx_ts[0]) & (df_col.index <= idx_ts[-1])]
        for col in df_col:
            df.loc[:, [col]] = df_col[col].values

    # for col in data['pharmaid'].unique():
    #     try:
    #         # if col == 1000974:
    #         #     pass
    #         # d = data[(data['pharmaid'] == col) & (data['givendose']!=0)].set_index('givenat')['givendose']
    #         d = data[(data['pharmaid'] == col) & (data['recordstatus'].isin(PHARMA_VALID))]
    #         if d.shape[0] == 0:
    #             continue
    #         d = d.resample(freq, origin=idx_ts[0]).sum()
    #         df_col = pd.DataFrame(index=d.index)
    #         df_col[col] = d.values
    #         df.update(df_col)
    #     except:
    #         print(f'Patient {data["patientid"][0]} - Pharma {col}')

    df.fillna(0, inplace=True)
    return df

