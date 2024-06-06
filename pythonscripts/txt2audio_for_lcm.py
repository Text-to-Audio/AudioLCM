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
from ldm.models.diffusion.scheduling_lcm import LCMSampler
from ldm.models.diffusion.plms import PLMSSampler
import pandas as pd
from torch.utils.data import DataLoader 
from tqdm import tqdm
from icecream import ic
from pathlib import Path
import soundfile as sf
import yaml
import datetime
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
        "--prompt_txt",
        type=str,
        nargs="?",
        default="prompt.txt",
        help="txt file with prompts in it"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default="22050",
        help="sample rate of wav"
    )
    parser.add_argument(
        "--inpaint",
        action='store_true',
        help="if test txt guided inpaint task"
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
        "--ddim_steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=20,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=312,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0, # if it's 1, only condition is taken into consideration
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
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
        default='vocoder/logs/audioset',
    )

    return parser.parse_args()

class GenSamples:
    def __init__(self,opt,sampler,model,outpath,vocoder = None,save_mel = True,save_wav = True, original_inference_steps=None) -> None:
        self.opt = opt
        self.sampler = sampler
        self.model = model
        self.outpath = outpath
        if save_wav:
            assert vocoder is not None
            self.vocoder = vocoder
        self.save_mel = save_mel
        self.save_wav = save_wav
        self.channel_dim = self.model.channels
        self.original_inference_steps = original_inference_steps
    
    def gen_test_sample(self,prompt,mel_name = None,wav_name = None):# prompt is {'ori_caption':’xxx‘,'struct_caption':'xxx'}
        uc = None
        record_dicts = []
        # if os.path.exists(os.path.join(self.outpath,mel_name+f'_0.npy')):
        #     return record_dicts
        if self.opt.scale != 1.0:
            emptycap = {'ori_caption':self.opt.n_samples*[""],'struct_caption':self.opt.n_samples*[""]}
            uc = self.model.get_learned_conditioning(emptycap)
        for n in range(self.opt.n_iter):# trange(self.opt.n_iter, desc="Sampling"):
            for k,v in prompt.items():
                prompt[k] = self.opt.n_samples * [v]
            c = self.model.get_learned_conditioning(prompt)# shape:[1,77,1280],即还没有变成句子embedding，仍是每个单词的embedding
            if self.channel_dim>0:
                shape = [self.channel_dim, self.opt.H, self.opt.W]  # (z_dim, 80//2^x, 848//2^x)
            else:
                shape = [self.opt.H, self.opt.W]
            samples_ddim, _ = self.sampler.sample(S=self.opt.ddim_steps,
                                                conditioning=c,
                                                batch_size=self.opt.n_samples,
                                                shape=shape,
                                                verbose=False,
                                                guidance_scale=self.opt.scale,
                                                original_inference_steps=self.original_inference_steps
                                                )
            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
            for idx,spec in enumerate(x_samples_ddim):
                spec = spec.squeeze(0).cpu().numpy()
                record_dict = {'caption':prompt['ori_caption'][0]}
                if self.save_mel:
                    mel_path = os.path.join(self.outpath,mel_name+f'_{idx}.npy')
                    np.save(mel_path,spec)
                    record_dict['mel_path'] = mel_path
                if self.save_wav:
                    wav = self.vocoder.vocode(spec)
                    wav_path = os.path.join(self.outpath,wav_name+f'_{idx}.wav')
                    soundfile.write(wav_path, wav, self.opt.sample_rate)
                    record_dict['audio_path'] = wav_path
                record_dicts.append(record_dict)
        return record_dicts


def main():
    opt = parse_args()

    config = OmegaConf.load(opt.base)

    # print("-------quick debug no load ckpt---------")
    # model = instantiate_from_config(config['model'])# for quick debug
    model = load_model_from_config(config, opt.resume)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = LCMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    if 'mel' in opt.vocoder_ckpt:
        vocoder = VocoderMelGan(opt.vocoder_ckpt,device)
    elif 'hifi' in opt.vocoder_ckpt:
        vocoder = VocoderHifigan(opt.vocoder_ckpt,device)
    elif 'bigv' in opt.vocoder_ckpt:
        vocoder = VocoderBigVGAN(opt.vocoder_ckpt,device)


    generator = GenSamples(opt,sampler,model,opt.outdir,vocoder,save_mel = False,save_wav = True, original_inference_steps=config.model.params.num_ddim_timesteps)
    csv_dicts = []
    
    with torch.no_grad():
        with model.ema_scope():
            if opt.test_dataset != 'none':
                if opt.test_dataset == 'audiocaps':
                    test_dataset = instantiate_from_config(config['test_dataset'])
                elif opt.test_dataset == 'clotho':
                    print("clotho!!!!!")
                    test_dataset = instantiate_from_config(config['test_dataset'])
                elif opt.test_dataset == 'fsd50k':
                    test_dataset = instantiate_from_config(config['test_dataset3'])
                print(f"Dataset: {type(test_dataset)} LEN: {len(test_dataset)}")

                import ipdb
                ipdb.set_trace()
                for item in tqdm(test_dataset):
                    prompt,f_name = item['caption'],item['f_name']
                    vname_num_split_index = f_name.rfind('_')# file_names[b]:video_name+'_'+num
                    v_n,num = f_name[:vname_num_split_index],f_name[vname_num_split_index+1:]
                    mel_name = f'{v_n}_sample_{num}'
                    wav_name = f'{v_n}_sample_{num}'

                    # write_gt_wav(v_n,opt.test_dataset2,opt.outdir,opt.sample_rate)
                    csv_dicts.extend(generator.gen_test_sample(prompt,mel_name=mel_name,wav_name=wav_name))

                df = pd.DataFrame.from_dict(csv_dicts)
                df.to_csv(os.path.join(opt.outdir,'result.csv'),sep='\t',index=False)
            else:
                with open(opt.prompt_txt,'r') as f:
                    prompts = f.readlines()

                for prompt in prompts:
                    wav_name = f'{prompt.strip().replace(" ", "-")}'
                    generator.gen_test_sample(prompt,wav_name=wav_name)

    print(f"Your samples are ready and waiting four you here: \n{opt.outdir} \nEnjoy.")

if __name__ == "__main__":
    main()

