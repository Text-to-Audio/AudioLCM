import pathlib
import sys
import os
directory = pathlib.Path(os.getcwd())
sys.path.append(str(directory))
import torch
import numpy as np
from wav_evaluation.models.CLAPWrapper import CLAPWrapper
import torch.nn.functional as F
import argparse
import csv
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import json





def cal_score_by_csv(csv_path,clap_model):    # audiocaps val的gt音频的clap_score计算为0.479077
    input_file = open(csv_path)
    input_lines = input_file.readlines()


        
    clap_scores = []

    caption_list,audio_list = [],[]
    with torch.no_grad():
        for idx in tqdm(range(len(input_lines))): 
            # text_embeddings = clap_model.get_text_embeddings([getattr(t,'caption')])# 经过了norm的embedding
            # audio_embeddings = clap_model.get_audio_embeddings([getattr(t,'audio_path')], resample=True)
            # score = clap_model.compute_similarity(audio_embeddings, text_embeddings,use_logit_scale=False)
            # clap_scores.append(score.cpu().numpy())
            if input_lines[idx][0] == 'S':
                item_name, semantic = input_lines[idx].split('\t')

                index = item_name[2:]
                # import ipdb
                # ipdb.set_trace()
                caption_list.append(semantic.strip())
                audio_list.append(f'/home1/liuhuadai/projects/VoiceLM-main/encodec_16k_6kbps_multiDisc/results/text_to_audio_0912/ref/{index}.wav')
            # import ipdb
            # ipdb.set_trace()
            if idx % 60 == 0:
                text_embeddings = clap_model.get_text_embeddings(caption_list)# 经过了norm的embedding
                audio_embeddings = clap_model.get_audio_embeddings(audio_list, resample=True)# 这一步比较耗时，读取音频并重采样到44100
                score_mat = clap_model.compute_similarity(audio_embeddings, text_embeddings,use_logit_scale=False)
                score = score_mat.diagonal()
                clap_scores.append(score.cpu().numpy())
                # print(caption_list)
                # print(audio_list)
                # print(score)
                audio_list = []
                caption_list = []
                # print("mean:",np.mean(np.array(clap_scores).flatten()))
    return np.mean(np.array(clap_scores).flatten())
[0.24463119, 0.24597324, 0.26050782, 0.25079757, 0.2501094, 0.2629509,0.25025588,0.25980043,0.27295044, 0.25655213, 0.2490872,  0.2598294,0.26491216,0.24698025,0.25086403,0.27533108,0.27969885,0.2596455,0.26313564,0.2658071]
def add_clap_score_to_csv(csv_path,clap_model):
    df = pd.read_csv(csv_path,sep='\t')
    clap_scores_dict = {}
    with torch.no_grad():
        for idx,t in enumerate(tqdm(df.itertuples()),start=1): 
            text_embeddings = clap_model.get_text_embeddings([getattr(t,'caption')])# 经过了norm的embedding
            audio_embeddings = clap_model.get_audio_embeddings([getattr(t,'audio_path')], resample=True)
            score = clap_model.compute_similarity(audio_embeddings, text_embeddings,use_logit_scale=False)
            clap_scores_dict[idx] = score.cpu().numpy()
    df['clap_score'] = clap_scores_dict
    df.to_csv(csv_path[:-4]+'_clap.csv',sep='\t',index=False)


if __name__ == '__main__':
    ckpt_path = '/home1/liuhuadai/projects/VoiceLM-main/encodec_16k_6kbps_multiDisc/useful_ckpts/CLAP'
    clap_model = CLAPWrapper(os.path.join(ckpt_path,'CLAP_weights_2022.pth'),os.path.join(ckpt_path,'config.yml'), use_cuda=True)

    clap_score = cal_score_by_csv('/home1/liuhuadai/projects/VoiceLM-main/encodec_16k_6kbps_multiDisc/Test/generate-test.txt',clap_model)
    out = 'text_to_audio2_0908'
    print(f"clap_score for {out} is:{clap_score}")
    print(f"clap_score for {out} is:{clap_score}")
    print(f"clap_score for {out} is:{clap_score}")
    # os.remove(csv_path)