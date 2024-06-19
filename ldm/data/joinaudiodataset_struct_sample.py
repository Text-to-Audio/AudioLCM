import sys
import numpy as np
import torch
import logging
import pandas as pd
import glob
logger = logging.getLogger(f'main.{__name__}')

sys.path.insert(0, '.')  # nopep8

class JoinManifestSpecs(torch.utils.data.Dataset):
    def __init__(self, split, main_spec_dir_path,other_spec_dir_path, mel_num=None, spec_crop_len=None,pad_value=-5,**kwargs):
        super().__init__()
        self.main_prob = 0.5
        self.split = split
        self.batch_max_length = spec_crop_len
        self.batch_min_length = 50
        self.mel_num = mel_num
        self.pad_value = pad_value
        manifest_files = []
        for dir_path in main_spec_dir_path.split(','):
            manifest_files += glob.glob(f'{dir_path}/*.tsv')
        df_list = [pd.read_csv(manifest,sep='\t') for manifest in manifest_files]
        self.df_main = pd.concat(df_list,ignore_index=True)

        manifest_files = []
        for dir_path in other_spec_dir_path.split(','):
            manifest_files += glob.glob(f'{dir_path}/*.tsv')
        df_list = [pd.read_csv(manifest,sep='\t') for manifest in manifest_files]
        self.df_other = pd.concat(df_list,ignore_index=True)

        if split == 'train':
            self.dataset = self.df_main.iloc[100:]
        elif split == 'valid' or split == 'val':
            self.dataset = self.df_main.iloc[:100]
        elif split == 'test':
            self.df_main = self.add_name_num(self.df_main)
            self.dataset = self.df_main
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
        if np.random.uniform(0,1) < self.main_prob:
            data = self.dataset.iloc[idx]
            ori_caption = data['ori_cap']
            struct_caption = data['caption']
        else:
            randidx = np.random.randint(0,len(self.df_other))
            data = self.df_other.iloc[randidx]
            ori_caption = data['caption']
            struct_caption = f'<{ori_caption}, all>'
        item = {}
        try:
            spec = np.load(data['mel_path']) # mel spec [80, 624]
        except:
            mel_path = data['mel_path']
            print(f'corrupted:{mel_path}')
            spec = np.ones((self.mel_num,self.batch_max_length)).astype(np.float32)*self.pad_value
        
        if spec.shape[1] <= self.batch_max_length:
            spec = np.pad(spec, ((0, 0), (0, self.batch_max_length - spec.shape[1])),mode='constant',constant_values = (self.pad_value,self.pad_value)) # [80, 624]

        item['image'] = spec[:self.mel_num,:self.batch_max_length]
        item["caption"] = {"ori_caption":ori_caption,"struct_caption":struct_caption}
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



