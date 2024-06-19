import os
import sys
import math
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed 
from typing import TypeVar, Optional, Iterator,List
import logging
import pandas as pd
import glob
import torch.distributed as dist
logger = logging.getLogger(f'main.{__name__}')

sys.path.insert(0, '.')  # nopep8

class JoinManifestSpecs(torch.utils.data.Dataset):
    def __init__(self, split, spec_dir_path, mel_num=80,spec_crop_len=1248,mode='pad',pad_value=-5,drop=0,**kwargs):
        super().__init__()
        self.split = split
        self.max_batch_len = spec_crop_len
        self.min_batch_len = 64
        self.mel_num = mel_num
        self.min_factor = 4
        self.drop = drop
        self.pad_value = pad_value
        assert mode in ['pad','tile']
        self.collate_mode = mode
        # print(f"################# self.collate_mode {self.collate_mode} ##################")

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

    def ordered_indices(self):
        index2dur = self.dataset[['duration']]
        index2dur = index2dur.sort_values(by='duration')
        return list(index2dur.index)
    
    def __getitem__(self, idx):
        item = {}
        data = self.dataset.iloc[idx]
        try:
            spec = np.load(data['mel_path']) # mel spec [80, 624]
        except:
            mel_path = data['mel_path']
            print(f'corrupted:{mel_path}')
            spec = np.ones((self.mel_num,self.min_batch_len)).astype(np.float32)*self.pad_value
        

        item['image'] = spec
        p = np.random.uniform(0,1)
        if p > self.drop:
            item["caption"] = data['caption']
        else:
            item["caption"] = ""
        if self.split == 'test':
            item['f_name'] = data['name']
        # item['f_name'] = data['mel_path']
        return item
    
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

        return to_dict

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

class JoinSpecsDebug(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('valid', **specs_dataset_cfg)
        self.dataset = self.dataset.iloc[:37]

class DDPIndexBatchSampler(Sampler):# 让长度相似的音频的indices合到一个batch中以避免过长的pad
    def __init__(self, indices ,batch_size, num_replicas: Optional[int] = None,
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
        self.indices = indices
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.batch_size = batch_size

        self.batches = self.build_batches()
        print(f"rank: {self.rank}, batches_num {len(self.batches)}")
        # If the dataset length is evenly divisible by replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.batches) % self.num_replicas != 0:
            self.batches = self.batches[:len(self.batches)//self.num_replicas*self.num_replicas]
        if len(self.batches) > self.num_replicas: 
            self.batches = self.batches[self.rank::self.num_replicas]
        else: # may happen in sanity checking
            self.batches = [self.batches[0]]
        print(f"after split batches_num {len(self.batches)}")
        self.shuffle = shuffle
        if self.shuffle:
            self.batches = np.random.permutation(self.batches)
        self.seed = seed

    def set_epoch(self,epoch):
        self.epoch = epoch
        if self.shuffle:
            np.random.seed(self.seed+self.epoch)
            self.batches = np.random.permutation(self.batches)

    def build_batches(self):
        batches,batch = [],[]
        for index in self.indices:
            batch.append(index)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        if not self.drop_last and len(batch) > 0:
            batches.append(batch)
        return batches    

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


def collate_1d_or_2d(values, pad_idx=0, left_pad=False, shift_right=False,min_len = None, max_len=None,min_factor=None, shift_id=1):
    if len(values[0].shape) == 1:
        return collate_1d(values, pad_idx, left_pad, shift_right,min_len, max_len,min_factor, shift_id)
    else:
        return collate_2d(values, pad_idx, left_pad, shift_right,min_len,max_len,min_factor)

def collate_1d(values, pad_idx=0, left_pad=False, shift_right=False,min_len=None, max_len=None,min_factor=None, shift_id=1):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    if max_len: 
        size = min(size,max_len)
    if min_len:
        size = max(size,min_len)
    if min_factor and (size % min_factor!=0):# size must be the multiple of min_factor
        size += (min_factor - size % min_factor)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), f"dst shape:{dst.shape} src shape:{src.shape}"
        if shift_right:
            dst[1:] = src[:-1]
            dst[0] = shift_id
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate_2d(values, pad_idx=0, left_pad=False, shift_right=False, min_len=None,max_len=None,min_factor=None):
    """Collate 2d for melspec,Convert a list of 2d tensors into a padded 3d tensor,pad in mel_length dimension. 
        values[0] shape: (melbins,mel_length)
    """
    size = max(v.shape[1] for v in values) # if max_len is None else max_len
    if max_len: 
        size = min(size,max_len)
    if min_len:
        size = max(size,min_len)
    if min_factor and (size % min_factor!=0):# size must be the multiple of min_factor
        size += (min_factor - size % min_factor)

    if isinstance(values,np.ndarray):
        values = torch.FloatTensor(values)
    if isinstance(values,list):
        values = [torch.FloatTensor(v) for v in values]
    res = torch.ones(len(values), values[0].shape[0],size).to(dtype=torch.float32)*pad_idx
    
    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), f"dst shape:{dst.shape} src shape:{src.shape}"
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v[:,:size], res[i][:,size - v.shape[1]:] if left_pad else res[i][:,:v.shape[1]])
    return res


def collate_1d_or_2d_tile(values, shift_right=False,min_len = None, max_len=None,min_factor=None, shift_id=1):
    if len(values[0].shape) == 1:
        return collate_1d_tile(values, shift_right,min_len, max_len,min_factor, shift_id)
    else:
        return collate_2d_tile(values, shift_right,min_len,max_len,min_factor)

def collate_1d_tile(values, shift_right=False,min_len=None, max_len=None,min_factor=None,shift_id=1):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    if max_len: 
        size = min(size,max_len)
    if min_len:
        size = max(size,min_len)
    if min_factor and (size%min_factor!=0):# size must be the multiple of min_factor
        size += (min_factor - size % min_factor)
    res = values[0].new(len(values), size)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), f"dst shape:{dst.shape} src shape:{src.shape}"
        if shift_right:
            dst[1:] = src[:-1]
            dst[0] = shift_id
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        n_repeat = math.ceil((size + 1) / v.shape[0])
        v = torch.tile(v,dims=(1,n_repeat))[:size]
        copy_tensor(v, res[i])

    return res


def collate_2d_tile(values, shift_right=False, min_len=None,max_len=None,min_factor=None):
    """Collate 2d for melspec,Convert a list of 2d tensors into a padded 3d tensor,pad in mel_length dimension. """
    size = max(v.shape[1] for v in values) # if max_len is None else max_len
    if max_len: 
        size = min(size,max_len)
    if min_len:
        size = max(size,min_len)
    if min_factor and (size % min_factor!=0):# size must be the multiple of min_factor
        size += (min_factor - size % min_factor)

    if isinstance(values,np.ndarray):
        values = torch.FloatTensor(values)
    if isinstance(values,list):
        values = [torch.FloatTensor(v) for v in values]
    res = torch.zeros(len(values), values[0].shape[0],size).to(dtype=torch.float32)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        n_repeat = math.ceil((size + 1) / v.shape[1])
        v = torch.tile(v,dims=(1,n_repeat))[:,:size]
        copy_tensor(v, res[i])
        
    return res