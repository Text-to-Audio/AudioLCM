import sys
import numpy as np
import torch
from typing import TypeVar, Optional, Iterator
import logging
import pandas as pd
from ldm.data.joinaudiodataset_anylen import *
import glob
logger = logging.getLogger(f'main.{__name__}')

sys.path.insert(0, '.')  # nopep8

class JoinManifestSpecs(torch.utils.data.Dataset):
    def __init__(self, split, main_spec_dir_path,other_spec_dir_path, mel_num=80,mode='pad', spec_crop_len=1248,pad_value=-5,drop=0,**kwargs):
        super().__init__()
        self.split = split
        self.max_batch_len = spec_crop_len
        self.min_batch_len = 64
        self.min_factor = 4
        self.mel_num = mel_num
        self.drop = drop
        self.pad_value = pad_value
        assert mode in ['pad','tile']
        self.collate_mode = mode
        manifest_files = []

        for dir_path in main_spec_dir_path.split(','):
            manifest_files += glob.glob(f'{dir_path}/*.tsv')
        df_list = [pd.read_csv(manifest,sep='\t') for manifest in manifest_files]
        self.df_main = pd.concat(df_list,ignore_index=True)

        manifest_files = []
        for dir_path in other_spec_dir_path.split(','):
            manifest_files += glob.glob(f'{dir_path}/*.tsv')
        df_list = [pd.read_csv(manifest,sep='\t') for manifest in manifest_files]
        # import ipdb
        # ipdb.set_trace()
        self.df_other = pd.concat(df_list,ignore_index=True)
        self.df_other.reset_index(inplace=True)

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
        print('dataset len:', len(self.dataset),"drop_rate",self.drop)

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
            df.loc[t[0],'name'] = str(df.loc[t[0],'name']) + f'_{t[1]}'
        return df

    def ordered_indices(self):
        index2dur = self.dataset[['duration']].sort_values(by='duration')
        index2dur_other = self.df_other[['duration']].sort_values(by='duration')
        other_indices = list(index2dur_other.index)
        offset = len(self.dataset)
        other_indices = [x + offset for x in other_indices]
        return list(index2dur.index),other_indices
        # return list(index2dur.index)

    def collater(self,inputs):
        to_dict = {}
        for l in inputs:
            for k,v in l.items():
                if k in to_dict:
                    to_dict[k].append(v)
                else:
                    to_dict[k] = [v]

        if self.collate_mode == 'pad':
            to_dict['image'] = collate_1d_or_2d(to_dict['image'],pad_idx=self.pad_value,min_len = self.min_batch_len,max_len=self.max_batch_len,min_factor=self.min_factor)
        elif self.collate_mode == 'tile':
            to_dict['image'] = collate_1d_or_2d_tile(to_dict['image'],min_len = self.min_batch_len,max_len=self.max_batch_len,min_factor=self.min_factor)
        else:
            raise NotImplementedError
        to_dict['caption'] = {'ori_caption':[c['ori_caption'] for c in to_dict['caption']],
                              'struct_caption':[c['struct_caption'] for c in to_dict['caption']]}

        return to_dict

    def __getitem__(self, idx):
        if idx < len(self.dataset):
            data = self.dataset.iloc[idx]
        # p = np.random.uniform(0,1)
        # if p > self.drop:
            ori_caption = data['ori_cap']
            struct_caption = data['caption']
        # else:
        #     ori_caption = ""
        #     struct_caption = ""
        else:
            data = self.df_other.iloc[idx-len(self.dataset)]
            # p = np.random.uniform(0,1)
            # if p > self.drop:
            ori_caption = data['caption']
            struct_caption = f'<{ori_caption}& all>'
            # else:
            #     ori_caption = ""
            #     struct_caption = ""
        item = {}
        try:
            spec = np.load(data['mel_path']) # mel spec [80, T]
            if spec.shape[1] > self.max_batch_len:
                spec = spec[:,:self.max_batch_len]
        except:
            mel_path = data['mel_path']
            print(f'corrupted:{mel_path}')
            spec = np.ones((self.mel_num,self.min_batch_len)).astype(np.float32)*self.pad_value
        
        item['image'] = spec
        item["caption"] = {"ori_caption":ori_caption,"struct_caption":struct_caption}
        if self.split == 'test':
            item['f_name'] = data['name']
        return item

    def __len__(self):
        return len(self.dataset) + len(self.df_other)
        # return len(self.dataset)


class JoinSpecsTrain(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('train', **specs_dataset_cfg)

class JoinSpecsValidation(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('valid', **specs_dataset_cfg)

class JoinSpecsTest(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('test', **specs_dataset_cfg)



class DDPIndexBatchSampler(Sampler):# 让长度相似的音频的indices合到一个batch中以避免过长的pad
    def __init__(self, main_indices,other_indices,batch_size, num_replicas: Optional[int] = None,
    # def __init__(self, main_indices,batch_size, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_initialized():
                # raise RuntimeError("Requires distributed package to be available")
                print("Not in distributed mode")
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_initialized():
                # raise RuntimeError("Requires distributed package to be available")
                rank = 0
            else:
                rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.main_indices = main_indices
        self.other_indices = other_indices
        self.max_index = max(self.other_indices)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches = self.build_batches()
        self.seed = seed

    def set_epoch(self,epoch):
        # print("!!!!!!!!!!!set epoch is called!!!!!!!!!!!!!!")
        self.epoch = epoch
        if self.shuffle:
            np.random.seed(self.seed+self.epoch)
            self.batches = self.build_batches()

    def build_batches(self):
        batches,batch = [],[]
        for index in self.main_indices:
            batch.append(index)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        if not self.drop_last and len(batch) > 0:
            batches.append(batch)
        selected_others = np.random.choice(len(self.other_indices),len(batches),replace=False)
        for index in selected_others:
            if index + self.batch_size > len(self.other_indices):
                index = len(self.other_indices) - self.batch_size
            batch = [self.other_indices[index + i] for i in range(self.batch_size)]
            batches.append(batch)
        self.batches = batches
        if self.shuffle:
            self.batches = np.random.permutation(self.batches)
        if self.rank == 0:
            print(f"rank: {self.rank}, batches_num {len(self.batches)}")

        if self.drop_last and len(self.batches) % self.num_replicas != 0:
            self.batches = self.batches[:len(self.batches)//self.num_replicas*self.num_replicas]
        if len(self.batches) >= self.num_replicas: 
            self.batches = self.batches[self.rank::self.num_replicas]
        else: # may happen in sanity checking
            self.batches = [self.batches[0]]
        if self.rank == 0:
            print(f"after split batches_num {len(self.batches)}")

        return self.batches

    def __iter__(self) -> Iterator[List[int]]:
        print(f"len(self.batches):{len(self.batches)}")
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)
