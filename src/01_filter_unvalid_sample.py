
import torch

from utils.config_dataset import *
from utils.ClassDataset import MergedDataset

from tqdm import tqdm

RANDOMSEED=2024
torch.manual_seed(RANDOMSEED)
np.random.seed(RANDOMSEED)



if __name__ == '__main__':
    print("Loading dataset")
    data_path = 'processed-merge/'
    pid_valid = pickle.load(open(os.path.join(data_path, 'pid_valid_00.p'), 'rb'))
    patient_info = pickle.load(open(os.path.join(data_path, 'patient_info.p'), 'rb'))
    sample_dict = pickle.load(open(os.path.join(data_path, 'sample_dict_vasopressor.p'), 'rb'))
    norm_params = pickle.load(open('processed-merge/norm_params_vasopressor.p', 'rb'))
    norm_params_info = pickle.load(open('processed-merge/norm_params_info_vasopressor.p', 'rb'))

    med_labels = np.array([sample_dict[i][0] for i in range(len(sample_dict))])
    selected_physio = ['HR', 'RR', 'SpO2', 'ABPd', 'ABPm', 'ABPs', 'ZVD']
    selected_med = ['norepinephrine', 'epinephrine', 'dobutamine']

    dataset = MergedDataset(
        sample_dict={item[0]: item[1] for item in sample_dict.items()},
        df_info=patient_info,
        norm=True, smooth=False, interpolate=False,
        selected_physio=selected_physio, selected_med=selected_med
    )


    for perc in [50, 60, 70, 80]:
        sample_dict_filtered = {}
        count_discard = 0
        idx_new = 0
        for idx, sample in enumerate(tqdm(dataset)):
            missing1 = np.isnan(sample['data'][:90, :]).sum() / 90 / 7
            missing2 = np.isnan(sample['data'][90:, :]).sum() / 90 / 7

            if missing1 <= 1 - perc/100 or missing2 <= 1 - perc/100:
                sample_dict_filtered[idx_new] = sample_dict[idx]
                idx_new += 1
            else:
                count_discard += 1
        print(f"Remained - {len(sample_dict_filtered)}\tDiscarded - {count_discard}")
        pickle.dump(sample_dict_filtered,
                    open(os.path.join(data_path, f'sample_dict_vasopressor_filtered{perc}.p'), 'wb'))