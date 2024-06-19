from glob import glob
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class TSVDataset(Dataset):
    def __init__(self, tsv_path, spec_crop_len=None):
        super().__init__()
        self.batch_max_length = spec_crop_len
        self.batch_min_length = 50
        df = pd.read_csv(tsv_path,sep='\t')
        df = self.add_name_num(df)
        self.dataset = df
        print('dataset len:', len(self.dataset))

    def add_name_num(self,df):
        """each file may have different caption, we add num to filename to identify each audio-caption pair"""
        name_count_dict = {}
        change = []
        for t in df.itertuples():
            name = getattr(t,'name')
            if name in name_count_dict:
                name_count_dict[name] += 1
            else:
                name_count_dict[name] = 0
            change.append((t[0],name_count_dict[name]))
        for t in change:
            df.loc[t[0],'name'] = df.loc[t[0],'name'] + f'_{t[1]}'
        return df


    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]
        item = {}
        spec = np.load(data['mel_path']) # mel spec [80, 624]
        if spec.shape[1] <= self.batch_max_length:
            spec = np.pad(spec, ((0, 0), (0, self.batch_max_length - spec.shape[1]))) # [80, 624]

        item['image'] = spec
        item["caption"] = data['caption']
        item["f_name"] = data['name']
        return item

    def __len__(self):
        return len(self.dataset)

class TSVDatasetStruct(TSVDataset):
    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]
        item = {}
        spec = np.load(data['mel_path']) # mel spec [80, 624]
        if spec.shape[1] <= self.batch_max_length:
            spec = np.pad(spec, ((0, 0), (0, self.batch_max_length - spec.shape[1]))) # [80, 624]

        item['image'] = spec[:,:self.batch_max_length]
        item["caption"] = {'ori_caption':data['ori_cap'],'struct_caption':data['caption']}
        item["f_name"] = data['name']
        return item

class TSVDatasetTestFake(TSVDataset):
    def __init__(self, specs_dataset_cfg):
        super().__init__(phase='test', **specs_dataset_cfg)
        self.dataset = [self.dataset[0]]




