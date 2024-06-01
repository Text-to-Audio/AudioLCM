import argparse, os, sys, glob
import pathlib
directory = pathlib.Path(os.getcwd())
print(directory)
sys.path.append(str(directory))
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import pandas as pd
from torch.utils.data import DataLoader 
from tqdm import tqdm
from icecream import ic
from pathlib import Path
import yaml
from vocoder.bigvgan.models import VocoderBigVGAN
import soundfile
# from pytorch_memlab import LineProfiler,profile

def load_model_from_config(config, ckpt = None, verbose=True):
    model = instantiate_from_config(config.model)
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]
        
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    else:
        print(f"Note chat no ckpt is loaded !!!")

    model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample_rate",
        type=int,
        default="16000",
        help="sample rate of wav"
    )

    parser.add_argument(
        "--test-dataset",
        default="none",
        help="test which dataset: audiocaps/clotho/fsd50k"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2audio-samples"
    )



    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default="",
    )
    parser.add_argument(
        "--vocoder-ckpt",
        type=str,
        help="paths to vocoder checkpoint",
        default='vocoder/logs/bigvnat16k93.5w',
    )

    return parser.parse_args()

class GenSamples:
    def __init__(self,opt,model,outpath,vocoder = None,save_mel = False,save_wav = True) -> None:
        self.opt = opt
        self.model = model
        self.outpath = outpath
        if save_wav:
            assert vocoder is not None
            self.vocoder = vocoder
        self.save_mel = save_mel
        self.save_wav = save_wav
    
    def gen_test_sample(self,mel,mel_name = None,wav_name = None):# prompt is {'ori_caption':’xxx‘,'struct_caption':'xxx'}
        uc = None
        record_dicts = []
        # if os.path.exists(os.path.join(self.outpath,mel_name+f'_0.npy')):
        #     return record_dicts
        # import ipdb
        # ipdb.set_trace()
        recon_mel,posterior = self.model(mel)
        spec = recon_mel.squeeze(0).cpu().numpy()

            
        if self.save_wav:
            wav = self.vocoder.vocode(spec)
            wav_path = os.path.join(self.outpath,wav_name+'.wav')
            soundfile.write(wav_path, wav, self.opt.sample_rate)
        return

def main():
    opt = parse_args()

    config = OmegaConf.load(opt.base)
    # print("-------quick debug no load ckpt---------")
    # model = instantiate_from_config(config['model'])# for quick debug
    model = load_model_from_config(config, opt.resume)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)


    os.makedirs(opt.outdir, exist_ok=True)
    if 'mel' in opt.vocoder_ckpt:
        vocoder = VocoderMelGan(opt.vocoder_ckpt,device)
    elif 'hifi' in opt.vocoder_ckpt:
        vocoder = VocoderHifigan(opt.vocoder_ckpt,device)
    elif 'bigv' in opt.vocoder_ckpt:
        vocoder = VocoderBigVGAN(opt.vocoder_ckpt,device)


    generator = GenSamples(opt,model,opt.outdir,vocoder,save_mel = False,save_wav = True)
    csv_dicts = []
    
    with torch.no_grad():
        if opt.test_dataset != 'none':
            if opt.test_dataset == 'audiocaps':
                test_dataset = instantiate_from_config(config['test_dataset'])
            elif opt.test_dataset == 'clotho':
                test_dataset = instantiate_from_config(config['test_dataset2'])
            elif opt.test_dataset == 'fsd50k':
                test_dataset = instantiate_from_config(config['test_dataset3'])
            elif opt.test_dataset == 'musiccap':
                test_dataset = instantiate_from_config(config['test_dataset'])
            print(f"Dataset: {type(test_dataset)} LEN: {len(test_dataset)}")
            for item in tqdm(test_dataset):
                mel,f_name = item['image'],item['f_name']
                mel = torch.from_numpy(mel).to(device).unsqueeze(0)
                vname_num_split_index = f_name.rfind('_')# file_names[b]:video_name+'_'+num
                v_n,num = f_name[:vname_num_split_index],f_name[vname_num_split_index+1:]
                mel_name = f'{v_n}_sample_{num}'
                wav_name = f'{v_n}_sample_{num}'
                generator.gen_test_sample(mel,mel_name=mel_name,wav_name=wav_name)
                # write_gt_wav(v_n,opt.test_dataset2,opt.outdir,opt.sample_rate)
                # csv_dicts.extend(generator.gen_test_sample(mel,mel_name=mel_name,wav_name=wav_name))

            # df = pd.DataFrame.from_dict(csv_dicts)
            # df.to_csv(os.path.join(opt.outdir,'result.csv'),sep='\t',index=False)
        else:
            with open(opt.prompt_txt,'r') as f:
                prompts = f.readlines()
            for prompt in prompts:
                wav_name = f'{prompt.strip().replace(" ", "-")}'
                generator.gen_test_sample(prompt,wav_name=wav_name)

    print(f"Your samples are ready and waiting four you here: \n{opt.outdir} \nEnjoy.")

if __name__ == "__main__":
    main()

