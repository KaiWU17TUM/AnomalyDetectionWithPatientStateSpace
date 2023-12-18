from utils.config_dataset import *
from utils.data_io import *

if __name__ == '__main__':
    pharmaid = 107
    toi = [60, 60]
    apache = 'Surgical Cardiovascular'

    data_path = f'sample_per_pharma/{apache.replace(" ", "")}/pharma{pharmaid}_{toi[0]}_{toi[1]}_{apache.replace(" ", "")}_normed.p'
    data_path = get_processed_path(data_path)

    dataset = HIRIDDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=1)

    for i_batch, sample_batched in enumerate(tqdm(dataloader)):
        print(1)



