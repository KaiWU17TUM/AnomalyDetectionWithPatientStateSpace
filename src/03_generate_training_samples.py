import os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from multiprocessing import Pool

import sys
sys.path.append(os.path.abspath('.'))
from  utils.config_dataset import *

os.chdir('/home/kai/DigitalICU/Experiments/HIRID-PatientStateSpace')


def save_data_sample(args):
    path_data_per_pat = args['path_data_per_pat']
    apache = args['apache']
    file = args['file']
    save_path = args['save_path']

    pid = os.path.splitext(file)[0]
    data = pd.read_csv(os.path.join(path_data_per_pat, apache, file), sep=',', header=[0, 1], index_col=[0])
    data.index = pd.to_datetime(data.index)

    df = pd.DataFrame(index=data.index)
    df[('pharma_impact', 0)] = 0

    data_ = data[[('pharma_mask', str(col)) for col in INPUT_OF_INTEREST]]
    check_pharma_exist = data_.sum().sum()
    if check_pharma_exist == 0:
        return 0
    df = pd.concat((df, data_), axis=1)

    med_valid = data_[data_.isin([1])].stack().index
    ts_med_valid = {}
    for ts, med in med_valid:
        t_start = ts - np.timedelta64(3, 'h')
        t_end = ts + np.timedelta64(3, 'h')
        if (t_start < df.index[0]) or (t_end > df.index[-1]):
            continue
        if med in ts_med_valid:
            ts_med_valid[med] += [ts]
        else:
            ts_med_valid[med] = [ts]
    if len(ts_med_valid) == 0:
        return 0

    mask_med = data.loc[:, data.columns.get_level_values(0) == 'pharma_mask']
    df.loc[mask_med.sum(axis=1) > 0, [('pharma_impact', 0)]] = 1
    # df = df.reset_index()
    # for idx, row in df[df['pharma_impact']>0].iterrows():
    #     if idx + 1 < len(df):
    #         if df.loc[idx, 'pharma_impact'] == df.loc[idx+1, 'pharma_impact']:
    #             continue
    #     if idx + 5 < len(df):
    #         idx_end = idx + 5
    #     else:
    #         idx_end = len(df)
    #     df.loc[idx:idx_end, 'pharma_impact'] = 1
    # df = df.set_index('index', drop=True)

    data_ = data[[('pharma', str(col)) for col in INPUT_OF_INTEREST]]
    df = pd.concat((df, data_), axis=1)
    data_ = data[[('physio_num', str(col)) for col in OUTPUT_OF_INTEREST]]
    df = pd.concat((df, data_), axis=1)

    for med in ts_med_valid.keys():
        tss = sorted(ts_med_valid[med])
        i = 1
        for j, ts in enumerate(tss):
            if (j > 0) and (ts - tss[j - 1] == np.timedelta64(2, 'm')):
                continue
            t_start = ts - np.timedelta64(3, 'h')
            t_end = ts + np.timedelta64(3, 'h')

            sample = df.loc[t_start:t_end, :]
            #sample.to_csv(os.path.join(save_path, med, f'{pid}_{i}.csv'))
            pickle.dump(sample, open(os.path.join(save_path, med, f'{pid}_{i}.p'), 'wb'))
            i += 1

    return 1


if __name__ == '__main__':

    path_root = '/home/kai/DigitalICU/Experiments/HIRID-PatientStateSpace/processed-v2/'
    path_data_per_pat = os.path.join(path_root, 'data_per_patient_resample2min')
    save_path = os.path.join(path_root, 'training_samples')

    Path(save_path).mkdir(exist_ok=True, parents=True)
    for med in INPUT_OF_INTEREST:
        subpath = os.path.join(save_path, str(med))
        Path(subpath).mkdir(exist_ok=True, parents=True)

    for apache in APACHE_OF_INTEREST:
        print(f'------------------------------------------------------------')
        print(f'------------{apache}---------------')
        print(f'------------------------------------------------------------')
        files = os.listdir(os.path.join(path_data_per_pat, apache))

        with Pool(48) as pool:
            for _ in tqdm(
                    pool.imap_unordered(
                        save_data_sample,
                        [
                            dict(
                            path_data_per_pat=path_data_per_pat,
                            apache=apache,
                            file=file,
                            save_path=save_path,
                            )
                            for file in files
                        ]
                    ), total=len(files)
            ):
                pass

    pid_group = pickle.load(open(os.path.join(path_processed, 'pid_group_valid.p'), 'rb'))
    pid_apache_dict = {}
    for apache in ['Surgical Cardiovascular', 'Cardiovascular', 'Pulmonary']:
        for pid in pid_group[apache]:
            pid_apache_dict[pid] = apache.replace(' ', '')

    train_stats = {}
    for med in INPUT_OF_INTEREST:
        train_stats[med] = {apache: [] for apache in APACHE_OF_INTEREST}

        files = os.listdir(os.path.join(save_path, str(med)))
        for file in tqdm(files):
            pid = int(file.split('_')[0])
            apache = pid_apache_dict[pid]
            train_stats[med][apache] += [pid]


    for med in train_stats:
        print(f'Pharma: {med}')
        for apache in train_stats[med]:
            print(f'\t{apache} --- {len(train_stats[med][apache])}')
        print('--------------------------------')
    print(111)
