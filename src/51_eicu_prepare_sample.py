import os
from pathlib import Path
import pickle

from utils.eicu import *



if __name__ == '__main__':
    save_path = 'processed-eicu'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    ######################################################################
    # Filter infusion table with valid entries
    ######################################################################
    print('Loading infusion table...')
    df_infusion = get_inf_table()

    print('Filtering data with valid infusion records...')
    # discard patient with invalid infusion records
    pid_all = df_infusion.patientunitstayid.unique()
    pid_valid = []
    for pid in tqdm(pid_all):
        flag_valid = True
        df = df_infusion[df_infusion.patientunitstayid == pid]

        if df.shape[0] == 0:
            continue

        for drug in df.drugname:
            if drug in vaso_invalid:
                flag_valid = False
                break
        if not flag_valid:
            continue

        if pd.isnull(df.drugrate).sum() > 0 or pd.isnull(df.drugamount).sum() \
                or pd.isnull(df.infusionrate).sum() > 0 or pd.isnull(df.volumeoffluid).sum() \
                or '' in df.drugrate.values or '' in df.drugamount.values \
                or '' in df.infusionrate.values or '' in df.volumeoffluid.values:
            continue

        pid_valid.append(pid)

    ######################################################################
    # filter patient by unitadmitsource and unitstaytype & with valid information
    ######################################################################
    print('Filtering data with valid patient information...')
    df_info = get_patient_info(pid_valid)
    df_info = df_info.loc[
        (df_info.unitadmitsource.isin(['Emergency Department', 'Direct Admit'])) &
        (df_info.unitstaytype == 'admit')
        ]
    df_info = df_info[df_info.gender != '']
    df_info = df_info.loc[(df_info.age != '') & (df_info.age != '> 89')]

    pid_valid = df_info.patientunitstayid.unique()

    ######################################################################
    # get prescription of vaso-active agents
    ######################################################################
    print('Loading medication table...')
    df_med = get_med_table(pid_valid)

    ######################################################################
    # select valid infusion segments
    ######################################################################
    print('Generating infusion samples...')
    samples_inf = []
    count = {med: [] for med in med_selected}
    for pid in tqdm(pid_valid):
        med_data = get_med_per_pat(df_med, pid)
        inf_data = get_inf_per_pat(df_infusion, pid)
        for med in med_selected:
            med_i = med_data[med_data[med]==1]
            inf_i = inf_data[med]
            if inf_i.shape[0] == 0 or med_i.shape[0] == 0:
                continue

            # take the first infusion with a corresponding prescription (administration time differs < 200 min)
            inf_start = inf_i['infusionoffset'].iloc[0]
            if inf_start < 0:
                continue
            med_i = med_i[med_i.drugstartoffset < inf_start]
            if med_i.shape[0] == 0:
                continue
            med_i = med_i.iloc[-1]
            if inf_start - med_i.drugstartoffset.item() > 200:
                continue

            sample = inf_i.loc[(inf_start - inf_i.infusionoffset >= 180) | (inf_i.infusionoffset - inf_start < 180)]
            count[med].append(sample.shape[0])

            samples_inf.append((pid, med, inf_start, sample))

    pickle.dump(samples_inf, open(os.path.join(save_path, 'infusion_samples.p'), 'wb'))

    for med in count:
        print(f"{med} ---")
        values, nums = np.unique(count[med], return_counts=True)
        for val, n in zip(values, nums):
            print(f"\t{val} - {n}")

    #
    check = [sp for _, sp in samples_inf if sp.shape[0]>10]
    check0 = [sp for _, sp in samples_inf if 0 in sp['drugrate'].astype(float)]

    ######################################################################
    # generate samples - infusion + physio
    ######################################################################

    samples = []
    for pid, med, inf_start, inf_data in samples_inf:



    print(111)

