import os
import numpy as np
import pandas as pd
import pickle

def read_reference_table(varref_path):
    """
    Read variableid-metavariableid mapping table for the merge step
    """
    STEPS_PER_HOURS = 60

    varref = pd.read_csv(varref_path, sep="\t", encoding='cp1252', index_col=0)

    pharmaref = varref[varref["type"] == "pharma"].rename(columns={"variableid": "pharmaid"})
    enum_ref = {'very short': int(STEPS_PER_HOURS / 12), 'short': 1 * STEPS_PER_HOURS, '4h': 4 * STEPS_PER_HOURS,
                '6h': 6 * STEPS_PER_HOURS, '12h': 12 * STEPS_PER_HOURS, '24h': 24 * STEPS_PER_HOURS,
                '3d': 72 * STEPS_PER_HOURS}
    pharmaref.loc[:, "pharmaactingperiod_min"] = pharmaref.pharmaactingperiod.apply(
        lambda x: enum_ref[x] if type(x) == str else np.nan)
    check_func = lambda x: float(x) if type(x)==float or "/" not in x else float(x.split("/")[0])/float(x.split("/")[1])
    pharmaref.loc[:, "unitconversionfactor"] = pharmaref.unitconversionfactor.apply(check_func)
    varref = varref[varref["type"] != "pharma"].copy()
    varref.drop(varref.index[varref.variableid.isnull()], inplace=True)
    varref.loc[:, "variableid"] = varref.variableid.astype(int)
    varref.set_index("variableid", inplace=True)
    return varref, pharmaref



path_processed = 'processed-v2/'
selected_physio = pd.read_csv(os.path.join(path_processed, 'selected_physio.csv'))
selected_pharma = pickle.load(open(os.path.join(path_processed, 'selected_pharma.p'), 'rb'))
varref, pharmaref = read_reference_table(os.path.join(path_processed, 'varref.tsv'))    # from HIRID GitHub repo

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

PHARMA_INFUSION_START = 524
PHARMA_INFUSION_END = 776
PHARMA_INFUSION_RECORD = [8, 520]
PHARMA_INFUSION = [PHARMA_INFUSION_START] + [PHARMA_INFUSION_END] + PHARMA_INFUSION_RECORD
PHARMA_INJECTION = [
    544,    # tablet --- givendose=0
    780,    # injection --- with givendose
]
PHARMA_INVALID = [[522, 526, 546, 782]]
PHARMA_VALID = PHARMA_INFUSION + PHARMA_INJECTION


# Selected input & output variables for training samples
INPUT_OF_INTEREST_BOLUS = [
    1000655,    # Adrenalin - bolus
    1000658,    # Noradrenalin - bolus
    1000747,    # Lasix - bolus: lower blood pressuuuure
]
INPUT_OF_INTEREST_INFUSION = [
    1000649,  # Adrenalin - infusion
    1000657,  # Noradrenalin - infusion
    426,  # Dobutrex - infusion: stimulate heart muscle
    # Sedation
    1000251,  # Fentanyl
    208,  # Disoprivan
    251,  # Dormicum
    1000659,  # Morphin
]
INPUT_OF_INTEREST = INPUT_OF_INTEREST_BOLUS + INPUT_OF_INTEREST_INFUSION

OUTPUT_OF_INTEREST = [
    1,      # SpO2
    2,      # respiration
    # 17,     # End tidal carbon dioxide concentration
    18,     # heart rate
    24,     # Invasive systolic arterial pressure
    25,     # Invasive diastolic arterial pressure
    26,     # Invasive mean arterial pressure
    34,     # Central venous pressure
    43,     # urine
    45,     # fluid intake
    46,     # fluid ouput
]

APACHE_OF_INTEREST = [
    'SurgicalCardiovascular',
    'Cardiovascular',
    'Pulmonary',
]