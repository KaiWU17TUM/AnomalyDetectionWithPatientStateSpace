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



# original data from HIRID
data_raw_path = 'physionet.org/files/hirid/1.1.1/raw_stage/observation_tables/parquet'
pharma_raw_path = 'physionet.org/files/hirid/1.1.1/raw_stage/pharma_records/parquet'
# data_merged_path = 'physionet.org/files/hirid/1.1.1/merged_stage/merged_stage_parquet'


# path_processed = 'processed-v2/'
# selected_physio = pd.read_csv(os.path.join(path_processed, 'selected_physio.csv'))
# selected_pharma = pickle.load(open(os.path.join(path_processed, 'selected_pharma.p'), 'rb'))
varref, pharmaref = read_reference_table(os.path.join('processed-merge-v3/', 'varref.tsv'))    # from HIRID GitHub repo

# pid_list = pickle.load(open('processed/pid_valid.p', 'rb'))
# pid_group = pickle.load(open('processed/pid_group_valid.p', 'rb'))
# patient_info = pickle.load(open('processed/patient_info_valid.p', 'rb'))
# pharma_data = pickle.load(open('processed/pharma_data_valid.p', 'rb'))
# patient_data = pickle.load(open('processed/patient_data_valid_with_uid.p', 'rb'))

# COL_INFO_NUM = ['age', 'los']
# COL_INFO_CAT = ['sex', 'discharge_status', 'APACHE']
# COL_PHARMA = list(set(selected_pharma['variableid'].tolist()))
# COL_PHYSIO_NUM = selected_physio.loc[selected_physio['type'] == 'n', 'uid'].unique().tolist()
# COL_PHYSIO_CAT = selected_physio.loc[selected_physio['type'] == 'c', 'uid'].unique().tolist()
# COL_PHYSIO_SETTING = selected_physio.loc[selected_physio['isSetting'] == 1, 'uid'].unique().tolist()
# COL_PHYSIO_FLUID = selected_physio.loc[selected_physio['category'] == 'Fluid-balance', 'uid'].unique().tolist()

PHARMA_INFUSION_START = 524
PHARMA_INFUSION_END = 776
PHARMA_INFUSION_RECORD = [8, 520]
PHARMA_INFUSION = [PHARMA_INFUSION_START] + [PHARMA_INFUSION_END] + PHARMA_INFUSION_RECORD
PHARMA_INJECTION = [
    544,    # tablet --- givendose=0
    780,    # injection --- with givendose
]
PHARMA_INVALID = [522, 526, 546, 782]
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

MEDICAL_RANGES = {
    'normal': {
        1: [90, 100],  # SpO2
        2: [12, 16],  # respiration
        18: [50, 100],  # heart rate
        24: [95, 145],  # Invasive systolic arterial pressure
        25: [60, 90],  # Invasive diastolic arterial pressure
        26: [70, 100],  # Invasive mean arterial pressure
        34: [3, 8],  # Central venous pressure
        43: [None, None],  # urine
        45: [None, None],  # fluid intake
        46: [25, 30],  # fluid ouput
    },
    'limit': {
        1: [40, 100],  # SpO2
        2: [0, 60],  # respiration
        18: [30, 400],  # heart rate
        24: [40, 300],  # Invasive systolic arterial pressure
        25: [20, 150],  # Invasive diastolic arterial pressure
        26: [30, 200],  # Invasive mean arterial pressure
        34: [0, 20],  # Central venous pressure
        43: [None, None],  # urine
        45: [None, None],  # fluid intake
        46: [0, 70],  # fluid ouput
    }
}

INFO_BENCHMARK = [
    'patientid',
    'datetime',
    'admissiontime',
    'age',
    'sex',
    'height',
]

MED_BENCHMARK = [
    'norepinephrine',
    'epinephrine',
    'dobutamine',
    'Loop diuretics',
    'Benzodiacepine',
    'Propofol',
    'Opiate',
]
MED_BENCHMARK_DICT_VID = {
    39: [1000462, 1000656, 1000657, 1000658],
    # norepinephrine
    40: [71, 1000649, 1000650, 1000655, 1000750],
    # epinephrine
    41: [426],
    # dobutamine
    69: [4, 482, 1000232, 1000520, 1000521, 1000522, 1000747, 1000986],
    # Loop diuretics
    77: [245, 246, 251, 252, 442, 1000239, 1000418, 1000700, 1000902, 1000976, 1000977, 1000978, 1000988, 1000991,
         1001051, 1001054, 1001215],
    # Benzodiacepine
    80: [208, 1000491, 1000691, 1000699, 1001050, 1001052, 1001053],
    # Propofol
    86: [204, 214, 215, 1000251, 1000350, 1000351, 1000382, 1000427, 1000444, 1000445, 1000446, 1000523, 1000627,
         1000659, 1000692, 1000705, 1000768, 1000771, 1000794, 1000862, 1000932, 1000933, 1000935, 1000936, 1000937,
         1000984, 1000985, 1000989, 1000990, 1001046],
    # Opiate
}

PHYSIO_BENCHMARK = [
    'HR',
    'RR',
    'SpO2',
    'ABPd',
    'ABPm',
    'ABPs',
    'ZVD',
    'IN',
    'OUT',
    'OUTurine/h',
]

PHYSIO_BENCHMARK_ALL = [
    'HR',
    'T Central',
    'ABPs',
    'ABPd',
    'ABPm',
    'NIBPs',
    'NIBPd',
    'NIBPm',
    'PAPm',
    'PAPs',
    'PAPd',
    'PCWP',
    'CO',
    'SvO2(m)',
    'ZVD',
    'ST1',
    'ST2',
    'ST3',
    'Rhythmus',
    'SpO2',
    'ETCO2',
    'RR',
    'supplemental oxygen',
    'OUTurine/h',
    'GCS Antwort',
    'GCS Motorik',
    'GCS Augen�ffnen',
    'RASS',
    'ICP',
    'TOF',
    'IN',
    'OUT',
    'Incrys',
    'Incolloid',
    'FIO2',
    'Peep',
    'Ventilator mode',
    'TV',
    'Spitzendruck',
    'Plateaudruck',
    'AWPmean',
    'RR set',
    'AiwayCode',
    'Haemofiltration',
    'Liquor/h',
    'Weight',
    'a-BE',
    'a_COHb',
    'a_Hb',
    'a_HCO3-',
    'a_Lac',
    'a_MetHb',
    'a_pH',
    'a_pCO2',
    'a_PO2',
    'a_SO2',
    'Zentral venöse sättigung',
    'Troponin-T',
    'creatine kinase',
    'creatine kinase-MB',
    'v-Lac',
    'BNP',
    'K+',
    'Na+',
    'Cl-',
    'Ca2+ ionizied',
    'Ca2+ total',
    'phosphate',
    'Mg_lab',
    'Urea',
    'creatinine',
    'urinary creatinin',
    'urinary Na+',
    'urinary urea',
    'ASAT',
    'ALAT',
    'bilirubine, total',
    'Bilirubin, direct',
    'alkaline phosphatase',
    'gamma-GT',
    'aPTT',
    'Fibrinogen',
    'FII',
    'Factor V',
    'Factor VII',
    'factor X',
    'INR',
    'albumin',
    'glucose',
    'Ammoniak',
    'C-reactive protein',
    'procalcitonin',
    'lymphocyte',
    'Neutr',
    'Segm. Neut.',
    'Stabk. Neut.',
    'BSR',
    'Hb',
    'total white blood cell count',
    'platelet count',
    'MCH',
    'MCHC',
    'MCV',
    'Ferritin',
    'TSH',
    'AMYL-S',
    'Lipase',
    'Cortisol',
    'pH Liquor',
    'Laktat Liquor',
    'Glucose Liquor',
    'pH Drain',
    'AMYL-Drainag',
]

PHYSIO_BENCHMARK_CAT = [
    'Rhythmus',
    'Ventilator mode',
    'AiwayCode',
]

NUM_CAT_PHYSIO_BENCHMARK = {
    'Rhythmus': 16,
    'Ventilator mode': 15,
    'AiwayCode': 6,
}



APACHE_BENCHMARK_MERGE = {
    'Cardiovascular':           [98, 190],
    'Respiratory':              [99, 191],
    'Gastrointestinal':         [100, 192],
    'Neurologic':               [101, 193],
    'Trauma':                   [103, 194],
    'Metabolic/Endocrinology':  [104, 195],
    'Hematologic':              [105, 196],
    'Other medical diseases':   [102, 106, 197, 198, 206],

    'Cardiovascular surgical':  [107, 199],
    'Respiratory surgical':     [108, 201],
    'Gastrointestinal surgical':[109, 200],
    'Neurologic surgical':      [110, 202],
    'Trauma surgical':          [111, 203],
    'Renal surgical':           [112, 204],
    'Other surgical':           [113, 114, 205]
}

APACHE_BENCHMARK_MERGE_INDEX = {
    k: i for i, k in enumerate(APACHE_BENCHMARK_MERGE)
}

SAMPLE_LEN_AE = 6 * 60   # minutes