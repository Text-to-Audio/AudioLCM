import pandas as pd
import audioread
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

def map_duration(tsv_withdur,tsv_toadd):# tsv_withdur 和 tsv_toadd 'name'列相同且tsv_withdur有duration信息，目标是给tsv_toadd的相同行加上duration信息。
    df1 = pd.read_csv(tsv_withdur,sep='\t')
    df2 = pd.read_csv(tsv_toadd,sep='\t')

    df = df2.merge(df1,on=['name'],suffixes=['','_y'])
    dropset = list(set(df.columns) - set(df1.columns))
    df = df.drop(dropset,axis=1)
    df.to_csv(tsv_toadd,sep='\t',index=False)
    return df

def add_duration(args):
    index,audiopath = args
    try:
        with audioread.audio_open(audiopath) as f:
            totalsec = f.duration
    except:
        totalsec = -1
    return (index,totalsec)

def add_dur2tsv(tsv_path,save_path):
    df = pd.read_csv(tsv_path,sep='\t')
    item_list = []
    for item in tqdm(df.itertuples()):
        item_list.append((item[0],getattr(item,'audio_path')))

    r = process_map(add_duration,item_list,max_workers=16,chunksize=32)
    index2dur = {}
    for index,dur in r:
        if dur == -1:
            bad_wav  = df.loc[index,'audio_path']
            print(f'bad wav:{bad_wav}')
        index2dur[index] = dur
        
    df['duration'] = df.index.map(index2dur)
    df.to_csv(save_path,sep='\t',index=False)

if __name__ == '__main__':
    add_dur2tsv('/root/autodl-tmp/liuhuadai/AudioLCM/now.tsv','/root/autodl-tmp/liuhuadai/AudioLCM/now_duration.tsv')
    #map_duration(tsv_withdur='tsv_maker/filter_audioset.tsv',
    #              tsv_toadd='MAA1 Dataset tsvs/V3/refilter_audioset.tsv')
