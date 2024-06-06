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




class GenSamples:
    def __init__(self,sampler,model,outpath,vocoder = None,save_mel = True,save_wav = True, original_inference_steps=None) -> None:
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
        emptycap = {'ori_caption':1*[""],'struct_caption':1*[""]}
        uc = self.model.get_learned_conditioning(emptycap)

        for n in range(1):# trange(self.opt.n_iter, desc="Sampling"):
            for k,v in prompt.items():
                prompt[k] = 1 * [v]
            c = self.model.get_learned_conditioning(prompt)# shape:[1,77,1280],即还没有变成句子embedding，仍是每个单词的embedding
            if self.channel_dim>0:
                shape = [self.channel_dim, 20, 312]  # (z_dim, 80//2^x, 848//2^x)
            else:
                shape = [20, 312]
            samples_ddim, _ = self.sampler.sample(S=2,
                                                conditioning=c,
                                                batch_size=1,
                                                shape=shape,
                                                verbose=False,
                                                guidance_scale=5,
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
                    soundfile.write(wav_path, wav, 16000)
                    record_dict['audio_path'] = wav_path
                record_dicts.append(record_dict)
        return record_dicts

def AudioLCMInfer(ori_prompt, config_path = "configs/audiolcm.yaml", model_path = "./model/000184.ckpt", vocoder_path = "./model/vocoder"):

    prompt = dict(ori_caption=ori_prompt,struct_caption=f'<{ori_prompt}& all>')


    config = OmegaConf.load(config_path)

    # print("-------quick debug no load ckpt---------")
    # model = instantiate_from_config(config['model'])# for quick debug
    model = load_model_from_config(config, model_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = LCMSampler(model)

    os.makedirs("results/test", exist_ok=True)

    vocoder = VocoderBigVGAN(vocoder_path,device)


    generator = GenSamples(sampler,model,"results/test",vocoder,save_mel = False,save_wav = True, original_inference_steps=config.model.params.num_ddim_timesteps)
    csv_dicts = []

    with torch.no_grad():
        with model.ema_scope():
                wav_name = f'{prompt["ori_caption"].strip().replace(" ", "-")}'
                generator.gen_test_sample(prompt,wav_name=wav_name)

    print(f"Your samples are ready and waiting four you here: \nresults/test \nEnjoy.")
    return "results/test/"+wav_name+"_0.wav"

def AudioLCMBatchInfer(ori_prompts, config_path = "configs/audiolcm.yaml", model_path = "./model/000184.ckpt", vocoder_path = "./model/vocoder"):

    prompts = [dict(ori_caption=ori_prompt,struct_caption=f'<{ori_prompt}& all>') for ori_prompt in ori_prompts]


    config = OmegaConf.load(config_path)

    # print("-------quick debug no load ckpt---------")
    # model = instantiate_from_config(config['model'])# for quick debug
    model = load_model_from_config(config, model_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = LCMSampler(model)

    os.makedirs("results/test", exist_ok=True)

    vocoder = VocoderBigVGAN(vocoder_path,device)


    generator = GenSamples(sampler,model,"results/test",vocoder,save_mel = False,save_wav = True, original_inference_steps=config.model.params.num_ddim_timesteps)
    csv_dicts = []

    for prompt in prompts:
        with torch.no_grad():
            with model.ema_scope():
                    wav_name = f'{prompt["ori_caption"].strip().replace(" ", "-")}'
                    generator.gen_test_sample(prompt,wav_name=wav_name)

    print(f"Your samples are ready and waiting four you here: \nresults/test \nEnjoy.")
    return "results/test/"+wav_name+"_0.wav"










