# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import getpass
import pdvega
from natsort import natsort_keygen
from IPython.display import display, HTML
from bigtree import list_to_tree, print_tree, preorder_iter
from tqdm import tqdm

# for configuring connection
from configobj import ConfigObj
import os

import warnings
warnings.filterwarnings('ignore')

# Connect to SQL database
def connect_to_sql():
    conn_info = {
        "sqluser": 'postgres',
        "sqlpass": 'postgres',
        "sqlhost": 'localhost',
        "sqlport": 5432,
        "dbname": 'eicu',
        "schema_name": 'public,eicu_crd',
    }
    print('Database: {}'.format(conn_info['dbname']))
    print('Username: {}'.format(conn_info["sqluser"]))

    con = psycopg2.connect(dbname=conn_info["dbname"],
                           host=conn_info["sqlhost"],
                           port=conn_info["sqlport"],
                           user=conn_info["sqluser"],
                           password=conn_info["sqlpass"])

    query_schema = 'set search_path to ' + conn_info['schema_name'] + ';'
    return con, query_schema

con, query_schema = connect_to_sql()


def unify_drugrate_unit(row):
    # used for pandas.Dataframe.apply()
    unit = row['drugname'].split('(')[1].split(')')[0]
    drugrate = row['drugrate']
    weight = row['patientweight']
    drugamount = row['drugamount']
    volume = row['volumeoffluid']
    drugrate_unified = unit_convert(drugrate, unit, weight, drugamount, volume)

    return drugrate_unified


def unit_convert(drugrate, unit, weight=None, drugamount=None, volume=None):
    if 'kg/' in unit:
        assert isinstance(weight, float), 'need patient weight to convert this unit.'
    if unit == 'ml/hr':
        assert isinstance(drugamount, float) & isinstance(volume, float), \
            'need infusion rate and fluid volume to convert this unit'
    if unit == 'mcg/min':
        rate = drugrate
    elif unit == 'mcg/kg/min':
        rate = drugrate * weight
    elif unit == 'mcg/hr':
        rate = drugrate / 60
    elif unit == 'mcg/kg/hr':
        rate = drugrate / 60 * weight
    elif unit == 'mg/min':
        rate = drugrate * 1000
    elif unit == 'mg/hr':
        rate = drugrate * 1000 / 60
    elif unit == 'mg/kg/min':
        rate = drugrate * 1000 * weight
    elif unit == 'ml/hr':
        rate = drugrate * drugamount / volume / 60
    elif unit == 'units/min':
        rate = drugrate
    else:
        assert ValueError(f'Unknown unit: {unit}')

    return rate



# Get patient info
def get_patient_info(pid):
    query = query_schema + """
    select *
    from patient
        where patientunitstayid in {} 
    order by patientunitstayid
    """.format(tuple(pid))

    df_info = pd.read_sql_query(query, con)
    return df_info


# Get vaso-active agents from medication table (prescription)
def get_med_table(pid):
    query = query_schema + """
    select *
    from medication
        where patientunitstayid in {} 
    order by patientunitstayid
    """.format(tuple(pid))

    df_med = pd.read_sql_query(query, con)
    df_med = df_med[df_med['drugordercancelled']=='No']
    # print(df_med.patientunitstayid.unique().shape)

    df_med_epinephrine = df_med.loc[
        (df_med.drughiclseqno.isin([37407, 39089, 36437, 34361, 2050])) |
        (pd.isnull(df_med.drughiclseqno) & df_med.drugname.str.lower().str.startswith('epinephrine'))
        ]
    df_med_epinephrine.insert(0, 'epinephrine', 1)
    df_med_epinephrine.insert(1, 'norepinephrine', 0)
    df_med_epinephrine.insert(2, 'dobutamine', 0)

    df_med_norepinephrine = df_med.loc[
        (df_med.drughiclseqno.isin([37410, 36346, 2051])) |
        (pd.isnull(df_med.drughiclseqno) & df_med.drugname.str.lower().str.contains('norepinephrine'))
        ]
    df_med_norepinephrine.insert(0, 'epinephrine', 0)
    df_med_norepinephrine.insert(1, 'norepinephrine', 1)
    df_med_norepinephrine.insert(2, 'dobutamine', 0)

    df_med_dobutamine = df_med.loc[
        (df_med.drughiclseqno.isin([8777, 40])) |
        (pd.isnull(df_med.drughiclseqno) & df_med.drugname.str.lower().str.contains('dobutamine')) |
        (pd.isnull(df_med.drughiclseqno) & df_med.drugname.str.lower().str.contains('dobutrex'))
        ]
    df_med_dobutamine.insert(0, 'epinephrine', 0)
    df_med_dobutamine.insert(1, 'norepinephrine', 0)
    df_med_dobutamine.insert(2, 'dobutamine', 1)

    df_med_vaso = pd.concat((df_med_epinephrine, df_med_norepinephrine, df_med_dobutamine), axis=0)
    df_med_vaso.sort_values(['patientunitstayid', 'drugstartoffset'], inplace=True)

    return df_med_vaso




# Get vaso-active agents from infusion table
vaso_dict = {
    'norepinephrine':[
        'Norepinephrine' ,
        'Norepinephrine ()' ,
        'Norepinephrine MAX 32 mg Dextrose 5% 250 ml (mcg/min)' ,
        'Norepinephrine MAX 32 mg Dextrose 5% 500 ml (mcg/min)' ,
        'Norepinephrine (mcg/hr)' ,
        'Norepinephrine (mcg/kg/hr)' ,
        'Norepinephrine (mcg/kg/min)' ,
        'Norepinephrine (mcg/min)' ,
        'Norepinephrine (mg/hr)' ,
        'Norepinephrine (mg/kg/min)' ,
        'Norepinephrine (mg/min)' ,
        'Norepinephrine (ml/hr)' ,
        'Norepinephrine STD 32 mg Dextrose 5% 282 ml (mcg/min)' ,
        'Norepinephrine STD 32 mg Dextrose 5% 500 ml (mcg/min)' ,
        'Norepinephrine STD 4 mg Dextrose 5% 250 ml (mcg/min)' ,
        'Norepinephrine STD 4 mg Dextrose 5% 500 ml (mcg/min)' ,
        'Norepinephrine STD 8 mg Dextrose 5% 250 ml (mcg/min)' ,
        'Norepinephrine STD 8 mg Dextrose 5% 500 ml (mcg/min)' ,
        'Norepinephrine (units/min)' ,
        'Norepinephrine (Unknown)' ,
        'norepinephrine Volume (ml)' ,
        'norepinephrine Volume (ml) (ml/hr)' ,
        'Levophed (mcg/kg/min)' ,
        'levophed  (mcg/min)' ,
        'levophed (mcg/min)' ,
        'Levophed (mcg/min)' ,
        'Levophed (mg/hr)' ,
        'levophed (ml/hr)' ,
        'Levophed (ml/hr)' ,
        'NSS with LEVO (ml/hr)' ,
        'NSS w/ levo/vaso (ml/hr)' ,
    ],
    'epinephrine':[
        'EPI (mcg/min)',
        'Epinepherine (mcg/min)',
        'Epinephrine',
        'Epinephrine ()',
        'EPINEPHrine(Adrenalin)MAX 30 mg Sodium Chloride 0.9% 250 ml (mcg/min)',
        'EPINEPHrine(Adrenalin)STD 4 mg Sodium Chloride 0.9% 250 ml (mcg/min)',
        'EPINEPHrine(Adrenalin)STD 4 mg Sodium Chloride 0.9% 500 ml (mcg/min)',
        'EPINEPHrine(Adrenalin)STD 7 mg Sodium Chloride 0.9% 250 ml (mcg/min)',
        'Epinephrine (mcg/hr)',
        'Epinephrine (mcg/kg/min)',
        'Epinephrine (mcg/min)',
        'Epinephrine (mg/hr)',
        'Epinephrine (mg/kg/min)',
        'Epinephrine (ml/hr)'
    ],
    'dobutamine':[
        'Dobutamine (mcg/kg/min)',
        'Dobutamine (mcg/min)',
        'Dobutamine (ml/hr)',
        'Dobutamine',
        'dobutrex (mg/kg/min)',
        'dobutrex (mcg/kg/min)',
        'dobutrex',
        'Dobutamine ()',
        'Dobutamine (mcg/kg/hr)',
        'Dobutamine (units/min)',
        'DOBUTamine STD 500 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)',
        'DOBUTamine MAX 1000 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)',
    ],
    'dopamine': [
        'Dopamine',
        'Dopamine ()',
        'DOPamine MAX 800 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)',
        'Dopamine (mcg/hr)',
        'Dopamine (mcg/kg/hr)',
        'dopamine (mcg/kg/min)',
        'Dopamine (mcg/kg/min)',
        'Dopamine (mcg/min)',
        'Dopamine (mg/hr)',
        'Dopamine (ml/hr)',
        'Dopamine (nanograms/kg/min)',
        'DOPamine STD 15 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)',
        'DOPamine STD 400 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)',
        'DOPamine STD 400 mg Dextrose 5% 500 ml  Premix (mcg/kg/min)',
        'Dopamine (Unknown)',
    ],
    'phenylephrine': [
        'Phenylephrine' ,
        'Phenylephrine ()' ,
        'Phenylephrine  MAX 100 mg Sodium Chloride 0.9% 250 ml (mcg/min)' ,
        'Phenylephrine (mcg/hr)' ,
        'Phenylephrine (mcg/kg/min)' ,
        'Phenylephrine (mcg/kg/min) (mcg/kg/min)' ,
        'Phenylephrine (mcg/min)' ,
        'Phenylephrine (mcg/min) (mcg/min)' ,
        'Phenylephrine (mg/hr)' ,
        'Phenylephrine (mg/kg/min)' ,
        'Phenylephrine (ml/hr)' ,
        'Phenylephrine  STD 20 mg Sodium Chloride 0.9% 250 ml (mcg/min)' ,
        'Phenylephrine  STD 20 mg Sodium Chloride 0.9% 500 ml (mcg/min)' ,
        'Volume (ml) Phenylephrine' ,
        'Volume (ml) Phenylephrine ()' ,
        'neo-synephrine (mcg/min)' ,
        'neosynephrine (mcg/min)' ,
        'Neosynephrine (mcg/min)' ,
        'Neo Synephrine (mcg/min)' ,
        'Neo-Synephrine (mcg/min)' ,
        'NeoSynephrine (mcg/min)' ,
        'NEO-SYNEPHRINE (mcg/min)' ,
        'Neosynephrine (ml/hr)' ,
        'neosynsprine' ,
        'neosynsprine (mcg/kg/hr)' ,
    ],
    'vasopressin':[
        'Vasopressin' ,
        'Vasopressin ()' ,
        'Vasopressin 20 Units Sodium Chloride 0.9% 100 ml (units/hr)' ,
        'Vasopressin 20 Units Sodium Chloride 0.9% 250 ml (units/hr)' ,
        'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/hr)' ,
        'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/kg/hr)' ,
        'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/min)' ,
        'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (Unknown)' ,
        'Vasopressin 40 Units Sodium Chloride 0.9% 200 ml (units/min)' ,
        'Vasopressin (mcg/kg/min)' ,
        'Vasopressin (mcg/min)' ,
        'Vasopressin (mg/hr)' ,
        'Vasopressin (mg/min)' ,
        'vasopressin (ml/hr)' ,
        'Vasopressin (ml/hr)' ,
        'Vasopressin (units/hr)' ,
        'Vasopressin (units/kg/min)' ,
        'vasopressin (units/min)' ,
        'Vasopressin (units/min)' ,
        'VAsopressin (units/min)' ,
        'Vasopressin (Unknown)' ,
    ],
    'milrinone':[
        'Milrinone' ,
        'Milrinone ()' ,
        'Milrinone (mcg/kg/hr)' ,
        'Milrinone (mcg/kg/min)' ,
        'Milrinone (ml/hr)' ,
        'Milrinone (Primacor) 40 mg Dextrose 5% 200 ml (mcg/kg/min)' ,
        'Milronone (mcg/kg/min)' ,
        'primacore (mcg/kg/min)' ,
    ],
    'heparin':[
        'Hepain (ml/hr)' ,
        'Heparin' ,
        'Heparin ()' ,
        'Heparin 25,000 Unit/D5w 250 ml (ml/hr)' ,
        'Heparin 25000 Units Dextrose 5% 500 ml  Premix (units/hr)' ,
        'Heparin 25000 Units Dextrose 5% 500 ml  Premix (units/kg/hr)' ,
        'Heparin 25000 Units Dextrose 5% 950 ml  Premix (units/kg/hr)' ,
        'HEPARIN #2 (units/hr)' ,
        'Heparin 8000u/1L NS (ml/hr)' ,
        'Heparin-EKOS (units/hr)' ,
        'Heparin/Femoral Sheath   (units/hr)' ,
        'Heparin (mcg/kg/hr)' ,
        'Heparin (mcg/kg/min)' ,
        'Heparin (ml/hr)' ,
        'heparin (units/hr)' ,
        'Heparin (units/hr)' ,
        'HEPARIN (units/hr)' ,
        'Heparin (units/kg/hr)' ,
        'Heparin (Unknown)' ,
        'Heparin via sheath (units/hr)' ,
        'Left  Heparin (units/hr)' ,
        'NSS carrier heparin (ml/hr)' ,
        'S-Heparin (units/hr)' ,
        'Volume (ml) Heparin-heparin 25,000 units in 0.45 % sodium chloride 500 mL infusion' ,
        'Volume (ml) Heparin-heparin 25,000 units in 0.45 % sodium chloride 500 mL infusion (ml/hr)' ,
        'Volume (ml) Heparin-heparin 25,000 units in dextrose 500 mL infusion' ,
        'Volume (ml) Heparin-heparin 25,000 units in dextrose 500 mL infusion (ml/hr)' ,
        'Volume (ml) Heparin-heparin infusion 2 units/mL in 0.9% sodium chloride (ARTERIAL LINE)' ,
        'Volume (ml) Heparin-heparin infusion 2 units/mL in 0.9% sodium chloride (ARTERIAL LINE) (ml/hr)' ,
    ]
}

vaso_all = [name for k in vaso_dict for name in vaso_dict[k]]
vaso_valid = [name for name in vaso_all
              if ('mcg/' in name) or ('mg/' in name)
              or ('ml/' in name) or ('units/' in name)]
vaso_invalid = [name for name in vaso_all if name not in vaso_valid]

med_selected = ['norepinephrine', 'epinephrine', 'dobutamine']
# med_selected = ['norepinephrine', 'epinephrine']   # CANNOT FIND DOBUTAMINE IN MEDICATION TABLE
vaso_selected = [name for name in vaso_valid for k in med_selected
                 if name in vaso_dict[k]]
vaso_unselected = [name for name in vaso_all if name not in vaso_selected]

vaso_dict_selected = {k:[] for k in med_selected}
for k in med_selected:
    for name in vaso_dict[k]:
        if name in vaso_selected:
            vaso_dict_selected[k].append(name)

def get_inf_table():
    query = query_schema + """
    select *
    from infusiondrug
        where drugname in {} 
    order by patientunitstayid, infusionoffset
    """.format(tuple(vaso_selected))

    df_infusion = pd.read_sql_query(query, con)

    return df_infusion


def get_med_per_pat(df_med, pid):
    df_med_epinephrine = df_med.loc[
        (df_med.patientunitstayid == pid) &
        ((df_med.drughiclseqno.isin([37407, 39089, 36437, 34361, 2050])) |
        (pd.isnull(df_med.drughiclseqno) & df_med.drugname.str.lower().str.startswith('epinephrine')))
        ]

    df_med_norepinephrine = df_med.loc[
        (df_med.patientunitstayid == pid) &
        ((df_med.drughiclseqno.isin([37410, 36346, 2051])) |
        (pd.isnull(df_med.drughiclseqno) & df_med.drugname.str.lower().str.contains('norepinephrine')))
        ]

    df_med_dobutamine = df_med.loc[
        (df_med.patientunitstayid == pid) &
        ((df_med.drughiclseqno.isin([8777, 40])) |
        (pd.isnull(df_med.drughiclseqno) & df_med.drugname.str.lower().str.contains('dobutamine')) |
        (pd.isnull(df_med.drughiclseqno) & df_med.drugname.str.lower().str.contains('dobutrex')))
        ]
    return {
        'norepinephrine': df_med_norepinephrine,
        'epinephrine': df_med_epinephrine,
        'dobutamine': df_med_dobutamine
    }


def get_inf_per_pat(df_infusion, pid):
    inf = df_infusion[df_infusion.patientunitstayid==pid]
    return {
        'norepinephrine': inf[inf.drugname.isin(vaso_dict_selected['norepinephrine'])],
        'epinephrine': inf[inf.drugname.isin(vaso_dict_selected['epinephrine'])],
        'dobutamine': inf[inf.drugname.isin(vaso_dict_selected['dobutamine'])]
    }

def infusion_info_helper(df):
    # df: dataframe of a single infusion of a patient
    df['inf_start'] = 0
    df['inf_end'] = 0
    df['infused'] = 0
    df['infused'][:-1] = (df['drugrate_unified'][:-1].astype(float) *
                     (df.infusionoffset.shift(-1)[:-1]-df.infusionoffset[:-1]))/1000
    df['infused_cumsum'] = df['infused'].cumsum()
    df['inf_left'] = 0

    return df


def generate_sample_physio(df_physio, sample_inf):
    samples = {med: [] for med in pid_sample}
    count_miss = 0

    for med in pid_sample:
        for pid, inf_start in tqdm(pid_sample[med]):
            #         med_ = df_med_vaso.loc[(df_med_vaso[med]==1) & (df_med_vaso.patientunitstayid==pid)]
            df_sample = df_physio.loc[
                (df_physio.patientunitstayid == pid) &
                (df_physio.observationoffset - inf_start < len_sample) &
                (inf_start - df_physio.observationoffset <= len_sample),
                ['patientunitstayid', 'observationoffset'] + physio_selected
            ].sort_values('observationoffset').reset_index(drop=True)

            if df_sample.shape[0] < 72 / 2:
                continue

            base_date = datetime.date(2000, 1, 1)
            inf_start_dt = datetime.datetime.combine(
                base_date + datetime.timedelta(int(inf_start / 60 // 24)),
                datetime.time(inf_start // 60 % 24, inf_start % 60)
            )
            new_index = pd.date_range(start=inf_start_dt - datetime.timedelta(hours=3),
                                      end=inf_start_dt + datetime.timedelta(hours=2, minutes=55),
                                      freq='5T')
            df = pd.DataFrame(columns=df_sample.columns, index=new_index)
            df['patientunitstayid'] = pid

            offset_dt = []
            for offset in df_sample['observationoffset']:
                dt = datetime.datetime.combine(
                    base_date + datetime.timedelta(int(offset / 60 // 24)),
                    datetime.time(offset // 60 % 24, offset % 60)
                )
                offset_dt.append(dt)

            df_sample['index'] = offset_dt
            df_sample.set_index('index', inplace=True)
            df_sample = df_sample.resample('5T', origin=inf_start_dt).last()

            df.update(df_sample)
            df = df.astype(float)

            if (np.isnan(df.iloc[:36]).sum() > 72 / 4).sum() > 1 \
                    or (np.isnan(df.iloc[36:]).sum() > 72 / 4).sum() > 1:
                count_miss += 1
                continue

            samples[med].append(df)

    print(count_miss)