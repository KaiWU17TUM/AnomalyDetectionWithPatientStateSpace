import gc
import numpy as np
import pandas as pd


def drop_duplicates_pharma(df):
    """
    df: long-format dataframe of a patient
    varref: variable reference table that contain the mean and standard deviation of values for a subset of variables
    """
    PHARMA_DATETIME, PHARMAID, INFID = 'givenat', 'pharmaid', 'infusionid'
    PHARMA_STATUS = 'recordstatus'
    PHARMA_VAL = 'givendose'
    INSTANTANEOUS_STATE = 780
    STOP_STATE = 776

    df_dup = df[df.duplicated([PHARMA_DATETIME, PHARMAID, INFID], keep=False)]
    for pharmaid in df_dup[PHARMAID].unique():
        for infusionid in df_dup[df_dup[PHARMAID] == pharmaid][INFID].unique():
            tmp = df_dup[(df_dup[PHARMAID] == pharmaid) & (df_dup[INFID] == infusionid)]
            if len(tmp[PHARMA_STATUS].unique()) == 1 and tmp[PHARMA_STATUS].unique()[0] == INSTANTANEOUS_STATE:
                for i in range(len(tmp)):
                    df.loc[tmp.index[i], INFID] = "%s_%s" % (int(df.loc[tmp.index[i], INFID]), i)
                # tmp = df[(df[PHARMAID] == pharmaid) & (
                #     df[INFID].apply(lambda x: "%s_" % (infusionid) in x if type(x) == str else False))]
            elif len(tmp[PHARMA_STATUS].unique()) == 1 and tmp[PHARMA_STATUS].unique()[0] == STOP_STATE:
                if (tmp[PHARMA_VAL] != 0).sum() == 1:
                    df.drop(tmp.index[tmp[PHARMA_VAL] == 0], inplace=True)
                else:
                    df.drop(tmp.index[:-1], inplace=True)
            elif len(tmp[PHARMA_STATUS].unique()) == 2 and STOP_STATE in tmp[PHARMA_STATUS].unique():
                df.drop(tmp.index[tmp[PHARMA_STATUS] != STOP_STATE], inplace=True)
            else:
                raise Exception("Debug needed")
    return df


# def transform_pharma_table_fn(pharma: pd.DataFrame, pharmaref, lst_pmid):
#     pharma_ids = set(pharmaref[PHARMAID].unique())
#     pharma = pharma.loc[pharma[PHARMAID].isin(pharma_ids)].copy()
#
#     pharma.drop(pharma.index[pharma[PHARMA_STATUS].isin(INVALID_PHARMA_STATUS)], inplace=True)
#     pharma.sort_values([PHARMAID, PHARMA_DATETIME, PHARMA_ENTERTIME], inplace=True)
#     pharma.loc[:, PHARMA_STATUS] = pharma[PHARMA_STATUS].replace(PSEUDO_INSTANTANEOUS_STATE, INSTANTANEOUS_STATE)
#     pharma = drop_duplicates_pharma(pharma)
#
#     if pharma.empty:
#         return pd.DataFrame()
#     else:
#         wide_pharma = []
#         for pharmaid in pharma[PHARMAID].unique():
#             pharma_acting_period = pharmaref[pharmaref[PHARMAID] == pharmaid].iloc[0].pharmaactingperiod_min
#             infusion_rate = []
#             for infusionid in pharma[pharma[PHARMAID] == pharmaid][INFID].unique():
#                 tmp_pharma = pharma[(pharma[PHARMAID] == pharmaid) & (pharma[INFID] == infusionid)].copy()
#                 infusion_rate.append(process_single_infusion(tmp_pharma, pharma_acting_period))
#             infusion_rate = pd.concat(infusion_rate, axis=1).sort_index()
#             infusion_rate = infusion_rate.sum(axis=1).to_frame(name="p%d" % pharmaid)
#             wide_pharma.append(infusion_rate)
#         wide_pharma = pd.concat(wide_pharma, axis=1).sort_index()
#         for pmid in lst_pmid:
#             cols = ['p%d' % x for x in pharmaref[pharmaref[METAVAR_ID] == pmid][PHARMAID]]
#             if np.isin(wide_pharma.columns, cols).sum() == 0:
#                 wide_pharma.loc[:, "pm%d" % pmid] = np.nan
#             else:
#                 if pharmaref[pharmaref[METAVAR_ID] == pmid][UNITCONVERT_FACTOR].notnull().sum() > 0:
#                     unitconverters = [pharmaref[(pharmaref[METAVAR_ID] == pmid) & (
#                             pharmaref[PHARMAID] == int(c[1:]))].iloc[0][UNITCONVERT_FACTOR] for c in
#                                       wide_pharma.columns[
#                                           np.isin(wide_pharma.columns, cols)]]
#                     wide_pharma.loc[:, "pm%d" % pmid] = (wide_pharma[
#                                                              wide_pharma.columns[np.isin(wide_pharma.columns,
#                                                                                          cols)]] * unitconverters).sum(
#                         axis=1)
#                 else:
#                     wide_pharma.loc[:, "pm%d" % pmid] = wide_pharma[
#                         wide_pharma.columns[np.isin(wide_pharma.columns, cols)]].sum(axis=1)
#                 wide_pharma.loc[wide_pharma.index[
#                                     wide_pharma[wide_pharma.columns[np.isin(wide_pharma.columns, cols)]].notnull().sum(
#                                         axis=1) == 0], "pm%d" % pmid] = np.nan
#                 wide_pharma.drop(wide_pharma.columns[np.isin(wide_pharma.columns, cols)], axis=1, inplace=True)
#
#         binary_pmids = ["pm%d" % x for x in
#                         pharmaref[pharmaref[METAVAR_UNIT].apply(lambda x: x == "Binary")][METAVAR_ID].values]
#         for col in binary_pmids:
#             wide_pharma.loc[:, col] = wide_pharma[col].apply(lambda x: x if np.isnan(x) else float(x != 0))
#
#         return wide_pharma
#


def resample_df(df, freq_string='2T'):
    cols = df.columns
    assert "datetime" in cols
    assert "patientid" in cols

    def reorder_time(patient_sample):
        pid = patient_sample["patientid"].iloc[0]

        patient_sample = patient_sample.reset_index(drop=True)
        HRs_non_zero = np.where(~np.isnan(patient_sample.HR))[0]
        if len(HRs_non_zero) > 0:

            HR_start_idx, HR_stop_idx = HRs_non_zero[0], HRs_non_zero[-1]

            patient_sample.loc[:HR_start_idx] = patient_sample.loc[:HR_start_idx].ffill()
        else:
            HR_start_idx, HR_stop_idx = 0, patient_sample.shape[0] - 1
        stay_stop_time, stay_start_time = patient_sample.loc[HR_stop_idx, "datetime"], patient_sample.loc[
            HR_start_idx, "datetime"]
        patient_sample = patient_sample.loc[HR_start_idx:HR_stop_idx].reset_index(drop=True)
        offset = np.timedelta64(stay_start_time.minute, 'm') \
                 + np.timedelta64(stay_start_time.second, 's') \
                 + np.timedelta64(stay_start_time.microsecond, 'us') + np.timedelta64(1, 'us')
        patient_sample.loc[:, "datetime"] = patient_sample["datetime"] - offset
        grided = patient_sample.set_index("datetime").resample(freq_string, axis=0, closed='left',
                                                                       label='right').last().reset_index()
        grided.loc[:, "datetime"] -= (stay_start_time - offset + np.timedelta64(1, 'us'))

        # grided = grided.reset_index(drop=True)
        grided["patientid"] = pid
        return grided

    dfs_pat = []
    for p in df["patientid"].unique():
        dfs_pat.append(reorder_time(df.query(f'{"patientid"} == {p}')))
        gc.collect()

    # df_part = pd.concat(dfs_pat).reset_index(drop=True)
    df_part = pd.concat(dfs_pat)
    df_part["patientid"] = df_part["patientid"].astype('int64')
    df_part = df_part[["patientid"] + [c for c in df_part.columns if c != "patientid"]]

    df_part["datetime"] /= np.timedelta64(60, 's')
    return df_part
