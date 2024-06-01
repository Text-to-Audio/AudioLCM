import pathlib
import sys
import os
directory = pathlib.Path(os.getcwd())
sys.path.append(str(directory))
import argparse
from wav_evaluation.metrics.fad import FrechetAudioDistance
"""it will resample to 16000hz automatically"""
def parse_args(): 
    parser = argparse.ArgumentParser()
    # parser.add_argument('--csv_path',type=str,default='tmp.csv')
    parser.add_argument('--pred_wavsdir',type=str)
    parser.add_argument('--gt_wavsdir', default="/home/tiger/nfs/data/audiocaps/test")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    frechet = FrechetAudioDistance(
        use_pca=False, 
        use_activation=False,
        verbose=False
    )
    fad_score = frechet.score(background_dir=args.gt_wavsdir,eval_dir=args.pred_wavsdir)
    print(f"Frechet Audio Distance {fad_score}")
