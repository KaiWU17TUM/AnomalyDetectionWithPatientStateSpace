import os
import numpy as np
import pandas as pd
import pickle



path_processed = 'processed-v2/'
selected_physio = pd.read_csv(os.path.join(path_processed, 'selected_physio.csv'))
selected_pharma = pickle.load(open(os.path.join(path_processed, 'selected_pharma.p'), 'rb'))

# pid_list = pickle.load(open('processed/pid_valid.p', 'rb'))
# pid_group = pickle.load(open('processed/pid_group_valid.p', 'rb'))
# patient_info = pickle.load(open('processed/patient_info_valid.p', 'rb'))
# pharma_data = pickle.load(open('processed/pharma_data_valid.p', 'rb'))
# patient_data = pickle.load(open('processed/patient_data_valid_with_uid.p', 'rb'))

COL_INFO_NUM = ['age', 'los']
COL_INFO_CAT = ['sex', 'discharge_status', 'APACHE']
COL_PHARMA = list(set(selected_pharma['variableid'].tolist()))
COL_PHYSIO_NUM = selected_physio.loc[selected_physio['type'] == 'n', 'uid'].unique().tolist()
COL_PHYSIO_CAT = selected_physio.loc[selected_physio['type'] == 'c', 'uid'].unique().tolist()
COL_PHYSIO_SETTING = selected_physio.loc[selected_physio['isSetting'] == 1, 'uid'].unique().tolist()
COL_PHYSIO_FLUID = selected_physio.loc[selected_physio['category'] == 'Fluid-balance', 'uid'].unique().tolist()