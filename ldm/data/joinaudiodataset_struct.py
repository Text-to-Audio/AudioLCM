import sys
import numpy as np
import torch
import logging
import pandas as pd
import glob
logger = logging.getLogger(f'main.{__name__}')

sys.path.insert(0, '.')  # nopep8

class JoinManifestSpecs(torch.utils.data.Dataset):
    def __init__(self, split, spec_dir_path, mel_num=None, spec_crop_len=None,drop=0,**kwargs):
        super().__init__()
        self.split = split
        self.batch_max_length = spec_crop_len
        self.batch_min_length = 50
        self.drop = drop
        self.mel_num = mel_num
        
        manifest_files = []
        for dir_path in spec_dir_path.split(','):
            manifest_files += glob.glob(f'{dir_path}/*.tsv')
        df_list = [pd.read_csv(manifest,sep='\t') for manifest in manifest_files]
        df = pd.concat(df_list,ignore_index=True)

        if split == 'train':
            self.dataset = df.iloc[100:]
        elif split == 'valid' or split == 'val':
            self.dataset = df.iloc[:100]
        elif split == 'test':
            df = self.add_name_num(df)
            self.dataset = df
        else:
            raise ValueError(f'Unknown split {split}')
        self.dataset.reset_index(inplace=True)
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
        try:
            spec = np.load(data['mel_path']) # mel spec [80, 624]
        except:
            mel_path = data['mel_path']
            print(f'corrupted:{mel_path}')
            spec = np.zeros((self.mel_num,self.batch_max_length)).astype(np.float32)
        
        if spec.shape[1] <= self.batch_max_length:
            spec = np.pad(spec, ((0, 0), (0, self.batch_max_length - spec.shape[1]))) # [80, 624]


        item['image'] = spec[:self.mel_num,:self.batch_max_length]
        p = np.random.uniform(0,1)
        if p > self.drop:
            item["caption"] = {"ori_caption":data['ori_cap'],"struct_caption":data['caption']}
        else:
            item["caption"] = {"ori_caption":"","struct_caption":""}
        
        if self.split == 'test':
            item['f_name'] = data['name']
        return item

    def __len__(self):
        return len(self.dataset)


class JoinSpecsTrain(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('train', **specs_dataset_cfg)

class JoinSpecsValidation(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('valid', **specs_dataset_cfg)

class JoinSpecsTest(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('test', **specs_dataset_cfg)



