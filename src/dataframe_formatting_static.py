import pickle
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import pandas as pd
import numpy as np


def percentile(n):
    def percentile_(x):
        return np.nanpercentile(x, n)

    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


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

if __name__=='__main__':
    #####################################################
    # dataset files
    #####################################################
    data_raw_path = '../physionet.org/files/hirid/1.1.1/raw_stage/observation_tables_parquet'
    data_merged_path = '../physionet.org/files/hirid/1.1.1/merged_stage/merged_stage_parquet'
    data_imputed_path = '../physionet.org/files/hirid/1.1.1/imputed_stage/imputed_stage_parquet'

    patient_info_file = '../physionet.org/files/hirid/1.1.1/reference_data/general_table.csv'
    variable_file = '../physionet.org/files/hirid/1.1.1/reference_data/hirid_variable_reference.csv'
    apache_group_file = '../processed/apache_groups.csv'

    variable_table = pd.read_csv(variable_file)
    selected_pharma = pd.read_csv('../processed/HiRID_selected_variables-input_dict.csv')
    selected_physio = pd.read_csv('../processed/HiRID_selected_variables-output_processed.csv')

    # uid_dict = {}
    # for index, row in selected_physio.iterrows():
    #     uid_dict[row['variableid']] = row['uid']

    pid_list = pickle.load(open('../processed/pid_valid.p', 'rb'))
    pid_group = pickle.load(open('../processed/pid_group_valid.p', 'rb'))
    patient_info = pickle.load(open('../processed/patient_info_valid.p', 'rb'))
    pharma_data = pickle.load(open('../processed/pharma_data_valid.p', 'rb'))

    patient_data_origin = pickle.load(open('../processed/patient_data_valid.p', 'rb'))
    patient_data = patient_data_origin.copy(deep=True)
    # patient_data['uid'] = 0.
    # for vid in tqdm(selected_physio['variableid'].unique()):
    #     uid = selected_physio.loc[selected_physio['variableid'] == vid, 'uid'].item()
    #     patient_data.loc[patient_data['variableid'] == vid, 'uid'] = uid
    # pickle.dump(patient_data, open('../processed/patient_data_valid_with_uid.p', 'wb'))
    patient_data = pickle.load(open('../processed/patient_data_valid_with_uid.p', 'rb'))

    # selected_pharma = selected_pharma.drop(
    #     selected_pharma[~selected_pharma['variableid'].isin(pharma_data['pharmaid'].unique())].index
    # ).reset_index(drop=True)
    #
    # COL_INFO_NUM = ['age', 'los']
    # COL_INFO_CAT = ['sex', 'discharge_status', 'APACHE']
    # COL_PHARMA = selected_pharma['variableid'].tolist()
    # COL_PHYSIO_NUM = selected_physio.loc[selected_physio['type'] == 'n', 'variableid'].tolist()
    # COL_PHYSIO_CAT = selected_physio.loc[selected_physio['type'] == 'c', 'variableid'].tolist()
    # COL_PHYSIO_SETTING = selected_physio.loc[selected_physio['isSetting'] == 1, 'variableid'].tolist()
    # COL_PHYSIO_FLUID = selected_physio.loc[selected_physio['category'] == 'Fluid-balance', 'uid'].unique().tolist()
    #
    # # ####################################################
    # # formated dataframe - patient information
    # # ####################################################
    # df_info = pd.DataFrame()
    # df_info = pd.concat((df_info, patient_info['patientid']), axis=1)
    # for col in COL_INFO_NUM:
    #     df_info = pd.concat((df_info, patient_info[col]), axis=1)
    # for col in COL_INFO_CAT:
    #     df_info = pd.concat((df_info, pd.get_dummies(patient_info[col], prefix=col)), axis=1)
    # df_info['los'] = df_info['los'].map(lambda x: x.days + x.seconds / 3600 / 24)
    # pickle.dump(df_info, open('../processed/df_info.p', 'wb'))
    #
    #
    # # ####################################################
    # # formated dataframe - pharma data
    # # ####################################################
    # df_pharma = pd.DataFrame(columns=['patientid'] + COL_PHARMA)
    # df_pharma['patientid'] = patient_info['patientid'].values
    # for pid in tqdm(pid_list):
    #     for pharma in COL_PHARMA:
    #         data = pharma_data.loc[(pharma_data['patientid'] == pid) & (pharma_data['pharmaid'] == pharma)]
    #         dosage = data['givendose'].sum()
    #         #         print(f"{pid} - {pharma}: {dosage}")
    #         df_pharma.loc[(df_pharma['patientid'] == pid), pharma] = dosage
    #
    # # ####################################################
    # # formated dataframe - patient data
    # # ####################################################
    # # Physio features for individual patients
    # gb_pid_physioid = patient_data.groupby(['patientid', 'uid']).agg(
    #     {'value': ['median', 'mad',
    #                percentile(1), percentile(5), percentile(25),
    #                percentile(75),percentile(95), percentile(99),
    #                ]
    #     }
    # )
    # # global physio statistics
    # gb_physioid = patient_data.groupby(['uid']).agg(
    #     {'value': ['median', 'mad',
    #                percentile(1), percentile(5), percentile(25),
    #                percentile(75),percentile(95), percentile(99),
    #                ]
    #     }
    # )
    # pickle.dump(gb_physioid, open('../processed/groupby_physioid.p', 'wb'))
    # pickle.dump(gb_pid_physioid, open('../processed/groupby_pid_physioid.p', 'wb'))
    #
    # # gb_physioid = pickle.load(open('../processed/groupby_physioid.p', 'rb'))
    # # gb_pid_physioid = pickle.load(open('../processed/groupby_pid_physioid.p', 'rb'))
    #
    # # ####################################################
    # # numeric physio data
    # col_num = ['patientid'] + [f'{uid}_{feat}'
    #            for uid in set(COL_PHYSIO_NUM).difference(set(COL_PHYSIO_FLUID))
    #            for feat in ['median', 'mad', '1', '5', '25', '75', '95', '99']]
    # df_physio = pd.DataFrame(columns=col_num)
    #
    # for pid in tqdm(pid_list):
    #     row = pd.DataFrame()
    #     row['patientid'] = [pid]
    #     for uid in set(COL_PHYSIO_NUM).difference(set(COL_PHYSIO_FLUID)):
    #         try:
    #             feat = gb_pid_physioid.loc[(pid, uid), 'value'].values
    #         except:
    #             feat = [np.nan]*8
    #
    #         cols = [f'{uid}_{x}' for x in ['median', 'mad', '1', '5', '25', '75', '95', '99']]
    #         feat_dict = {col: [feat[i]] for i, col in enumerate(cols)}
    #         feat_row = pd.DataFrame.from_dict(feat_dict)
    #
    #         row = pd.concat((row, feat_row), axis=1)
    #     df_physio =pd.concat((df_physio,row), axis=0)
    # pickle.dump(df_physio, open("../processed/df_physio1.p", "wb"))
    #
    # # ####################################################
    # df_physio = pickle.load(open("../processed/df_physio1.p", "rb"))
    # # categorical physio data
    # for uid in COL_PHYSIO_CAT:
    #     val_cat = sorted(patient_data[patient_data['uid']==uid]['value'].astype(int).unique())
    #     col_cat = [f'{uid}_{val}' for val in val_cat]
    #     df_cat = pd.DataFrame(columns=col_cat)
    #
    #     for pid in tqdm(pid_list):
    #         data =  patient_data.loc[(patient_data['uid']==uid) & (patient_data['patientid']==pid)]
    #         row = pd.DataFrame.from_dict({col: [0] for col in col_cat})
    #         if data.shape[0] == 0:
    #             pass
    #         else:
    #             row_count = pd.get_dummies(data['value'].astype(int), prefix=uid).sum().to_frame().T
    #             row.update(row_count)
    #         df_cat = pd.concat((df_cat, row), axis=0)
    #
    #     df_physio = pd.concat((df_physio, df_cat), axis=1)
    # pickle.dump(df_physio, open("../processed/df_physio2.p", "wb"))
    #
    # # ####################################################
    # # Fluid data
    # col_fluid = ['patientid', 'urine', 'drain', 'intake']
    # df_physio_fluid = pd.DataFrame(columns=col_fluid)
    # thres_urine = np.percentile(patient_data.loc[patient_data['uid']==31, 'value'], 99.95)
    # thres_intake = np.percentile(patient_data.loc[patient_data['uid']==33, 'value'], 99.95)
    # for pid in tqdm(pid_list):
    #     # 31 - Urine[ml|h]: total amount
    #     urine = patient_data.loc[(patient_data['uid'] == 31)
    #                              & (patient_data['patientid'] == pid)
    #                              & (patient_data['value']< thres_urine)].sort_values('datetime').reset_index(drop=True)
    #     if urine.shape[0] > 0:
    #         t_start = urine['datetime'][0]
    #         if urine[urine['value']>thres_urine].shape[0] > 0:
    #             print(f"Abnormal urine value: {urine.loc[urine['value']>thres_urine, 'value'].tolist()}")
    #         # dt = [urine['datetime'].iloc[0] - admtime] + urine['datetime'].diff()[1:].tolist()
    #         dt = urine['datetime'].diff()[1:].tolist()  # not sure what is the starting timestamp for the first measurement
    #         dt_h = np.array([item.days * 24 + item.seconds / 3600 for item in dt])
    #         dv = urine['value'].to_numpy()[1:]
    #         val_urine = (dt_h * dv).sum()
    #     else:
    #         val_urine = 0
    #
    #     # 32 - Drain [ml]: total amount
    #     if 't_start' in locals():
    #         drain = patient_data.loc[(patient_data['uid'] == 32)
    #                                  & (patient_data['patientid'] == pid)
    #                                  & (patient_data['datetime']>=t_start)]
    #     else:
    #         drain = patient_data.loc[(patient_data['uid'] == 32)
    #                                  & (patient_data['patientid'] == pid)]
    #     if drain.shape[0] > 0:
    #         val_drain = drain['value'].sum().item()
    #     else:
    #         val_drain = 0
    #
    #     # 33 - Fluid balance [ml]: total amount (last entry)
    #     intake = patient_data.loc[(patient_data['uid'] == 33)
    #                               & (patient_data['patientid'] == pid)
    #                               & (patient_data['value']<thres_intake)].sort_values('datetime').reset_index(drop=True)
    #     try:
    #         val_before_t_start = intake.loc[intake['datetime']<t_start, 'value'].iloc[-1]
    #     except:
    #         val_before_t_start = 0
    #     if intake.shape[0] > 0:
    #         if intake[intake['value'] > thres_intake].shape[0] > 0:
    #             print(f"Abnormal urine value: {intake.loc[intake['value'] > thres_intake, 'value'].tolist()}")
    #         val_intake = 0
    #         gb_intake = intake.groupby('variableid')
    #         for key in gb_intake.groups.keys():
    #             gdata = gb_intake.get_group(key)
    #             # a patient can get multiple infusion of the same variableid, check if there is renewal of the same medicine
    #             val_intake += gdata.shift(1)[gdata['value'].diff().lt(0)]['value'].sum()
    #             val_intake += gdata['value'].iloc[-1]
    #         val_intake -= val_before_t_start
    #     else:
    #         val_intake = 0
    #
    #     row = pd.DataFrame.from_dict({col: [val] for col, val in zip(col_fluid, [pid, val_urine, val_drain, val_intake])})
    #     df_physio_fluid = pd.concat((df_physio_fluid, row), axis=0)
    # pickle.dump(df_physio_fluid, open("../processed/df_physio3.p", "wb"))
    #
    # # ####################################################
    # # merge_all_columnes
    # # df_physio_fluid = pickle.load(open("../processed/df_physio3.p", "rb")).reset_index(drop=True)
    # df_physio_fluid['fluid_balance'] = df_physio_fluid['intake'] - df_physio_fluid['urine'] - df_physio_fluid['drain']
    #
    # # df_physio = pickle.load(open("../processed/df_physio2.p", "rb")).reset_index(drop=True)
    # df_physio = df_physio.merge(df_physio_fluid, on='patientid')
    # pickle.dump(df_physio, open("../processed/df_physio.p", "wb"))
    #
    #
    #
    #
    # print(111)