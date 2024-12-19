from preprocess.NAT_mel import MelNet
import os
from tqdm import tqdm
from glob import glob
import math
import pandas as pd
import argparse
from argparse import Namespace
import math
import audioread
from tqdm.contrib.concurrent import process_map
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torch.distributed import init_process_group
from torch.utils.data import Dataset,DataLoader,DistributedSampler
import torch.multiprocessing as mp
import json


class tsv_dataset(Dataset):
    def __init__(self,tsv_path,sr,mode='none',hop_size = None,target_mel_length = None) -> None:
        super().__init__()
        if os.path.isdir(tsv_path):
            files = glob(os.path.join(tsv_path,'*.tsv'))
            df = pd.concat([pd.read_csv(file,sep='\t') for file in files])
        else:
            df = pd.read_csv(tsv_path,sep='\t')
        self.audio_paths = []
        self.sr = sr
        self.mode = mode
        self.target_mel_length = target_mel_length
        self.hop_size = hop_size
        for t in tqdm(df.itertuples()):
            self.audio_paths.append(getattr(t,'audio_path'))

    def __len__(self):
        return len(self.audio_paths)

    def pad_wav(self,wav):
        # wav should be in shape(1,wav_len)
        wav_length = wav.shape[-1]
        assert wav_length > 100, "wav is too short, %s" % wav_length
        segment_length = (self.target_mel_length + 1) * self.hop_size  # final mel will crop the last mel, mel = mel[:,:-1]
        if segment_length is None or wav_length == segment_length:
            return wav
        elif wav_length > segment_length:
            return wav[:,:segment_length]
        elif wav_length < segment_length:
            temp_wav = torch.zeros((1, segment_length),dtype=torch.float32)
            temp_wav[:, :wav_length] = wav
        return temp_wav

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        wav, orisr = torchaudio.load(audio_path)
        if wav.shape[0] != 1: # stereo to mono  (2,wav_len) -> (1,wav_len)
            wav = wav.mean(0,keepdim=True)
        wav = torchaudio.functional.resample(wav, orig_freq=orisr, new_freq=self.sr)
        if self.mode == 'pad':
            assert self.target_mel_length is not None
            wav = self.pad_wav(wav)
        return audio_path,wav

def process_audio_by_tsv(rank,args):
    if args.num_gpus > 1:
        init_process_group(backend=args.dist_config['dist_backend'], init_method=args.dist_config['dist_url'],
                            world_size=args.dist_config['world_size'] * args.num_gpus, rank=rank)
    
    sr = args.audio_sample_rate
    dataset = tsv_dataset(args.tsv_path,sr = sr,mode=args.mode,hop_size=args.hop_size,target_mel_length=args.batch_max_length)
    sampler = DistributedSampler(dataset,shuffle=False) if args.num_gpus > 1 else None
    # batch_size must == 1,since wav_len is not equal
    loader = DataLoader(dataset, sampler=sampler,batch_size=1, num_workers=16,drop_last=False)

    device = torch.device('cuda:{:d}'.format(rank))
    mel_net = MelNet(args.__dict__)
    mel_net.to(device)
    # if args.num_gpus > 1: # RuntimeError: DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
    #     mel_net = DistributedDataParallel(mel_net, device_ids=[rank]).to(device)
    root = args.save_path
    loader = tqdm(loader) if rank == 0 else loader
    for batch in loader:
        audio_paths,wavs = batch
        wavs = wavs.to(device)
        if args.save_resample:               
            for audio_path,wav in zip(audio_paths,wavs):
                psplits = audio_path.split('/')
                wav_name = psplits[-1]
                # save resample
                resample_root,resample_name = root+f'_{sr}',wav_name[:-4]+'_audio.npy'
                resample_dir_name = os.path.join(resample_root,*psplits[1:-1])
                resample_path = os.path.join(resample_dir_name,resample_name)
                os.makedirs(resample_dir_name,exist_ok=True)
                np.save(resample_path,wav.cpu().numpy().squeeze(0))  

        if args.save_mel:
            mode = args.mode
            batch_max_length = args.batch_max_length

            for audio_path,wav in zip(audio_paths,wavs):
                psplits = audio_path.split('/')
                wav_name = psplits[-1]
                mel_root,mel_name = root,wav_name[:-4]+'_mel.npy'
                mel_dir_name = os.path.join(mel_root,f'mel{mode}{sr}',*psplits[1:-1])
                mel_path = os.path.join(mel_dir_name,mel_name)
                if not os.path.exists(mel_path):
                    mel_spec = mel_net(wav).cpu().numpy().squeeze(0) # (mel_bins,mel_len) 
                    if mel_spec.shape[1] <= batch_max_length:
                        if mode == 'tile': # pad is done in dataset as pad wav
                            n_repeat = math.ceil((batch_max_length + 1) / mel_spec.shape[1])
                            mel_spec = np.tile(mel_spec,reps=(1,n_repeat))
                        elif mode == 'none' or mode == 'pad':
                            pass
                        else:
                            raise ValueError(f'mode:{mode} is not supported')
                    mel_spec = mel_spec[:,:batch_max_length]
                    os.makedirs(mel_dir_name,exist_ok=True)
                    np.save(mel_path,mel_spec)      


def split_list(i_list,num):
    each_num = math.ceil(i_list / num)
    result = []
    for i in range(num):
        s = each_num * i
        e = (each_num * (i+1))
        result.append(i_list[s:e])
    return result


def drop_bad_wav(item):
    index,path = item
    try:
        with audioread.audio_open(path) as f:
            totalsec = f.duration
            if totalsec < 0.1:
                return index # index
    except:
        print(f"corrupted wav:{path}")
        return index
    return False 

def drop_bad_wavs(tsv_path):# 'audioset.csv'
    df = pd.read_csv(tsv_path,sep='\t')
    item_list = []
    for item in tqdm(df.itertuples()):
        item_list.append((item[0],getattr(item,'audio_path')))

    r = process_map(drop_bad_wav,item_list,max_workers=16,chunksize=16)
    bad_indices = list(filter(lambda x:x!= False,r))
        
    print(bad_indices)
    with open('bad_wavs.json','w') as f:
        x = [item_list[i] for i in bad_indices]
        json.dump(x,f)
    df = df.drop(bad_indices,axis=0)
    df.to_csv(tsv_path,sep='\t',index=False)

def addmel2tsv(save_dir,tsv_path):
    df = pd.read_csv(tsv_path,sep='\t')
    mels = glob(f'{save_dir}/mel{args.mode}{args.audio_sample_rate}/**/*_mel.npy',recursive=True)
    name2mel,idx2name,idx2mel = {},{},{}
    for mel in mels:
        bn = os.path.basename(mel)[:-8]# remove _mel.npy
        name2mel[bn] = mel
    for t in df.itertuples():
        idx = int(t[0])
        bn = os.path.basename(getattr(t,'audio_path'))[:-4]
        idx2name[idx] = bn
    for k,v in idx2name.items():
        idx2mel[k] = name2mel[v]
    df['mel_path'] = df.index.map(idx2mel)
    df.to_csv(tsv_path,sep='\t',index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--tsv_path",type=str)
    parser.add_argument( "--num_gpus",type=int,default=1)
    parser.add_argument( "--max_duration",type=int,default=30)
    return parser.parse_args()

if __name__ == '__main__':
    pargs = parse_args()
    tsv_path = pargs.tsv_path
    if os.path.isdir(tsv_path):
        files = glob(os.path.join(tsv_path,'*.tsv'))
        for file in files:
            drop_bad_wavs(file)
    else:
        drop_bad_wavs(tsv_path)
    num_gpus = pargs.num_gpus
    batch_max_length = int(pargs.max_duration * 62.5)# 62.5 is the mel length for 1 second
    save_path = 'processed'
    args = {
        'audio_sample_rate': 16000,
        'audio_num_mel_bins':80,
        'fft_size': 1024,
        'win_size': 1024,
        'hop_size': 256,
        'fmin': 0,
        'fmax': 8000,
        'batch_max_length': batch_max_length, 
        'tsv_path': tsv_path,
        'num_gpus': num_gpus,
        'mode': 'none', # pad,none,
        'save_resample':False,
        'save_mel' :True,
        'save_path': save_path,
    }
    os.makedirs(save_path,exist_ok=True)
    args = Namespace(**args)  
    args.dist_config = {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54189",
        "world_size": 1
    }
    if args.num_gpus>1:
        mp.spawn(process_audio_by_tsv,nprocs=args.num_gpus,args=(args,))
    else:
        process_audio_by_tsv(0,args=args)
    print("proceoss mel done")
    addmel2tsv(save_path,tsv_path)
    print("done")
    
