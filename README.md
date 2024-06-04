# AudioLCM: Text-to-Audio Generation with Latent Consistency Models

#### Huadai Liu, Rongjie Huang, Yang Liu, Hengyuan Cao, Jialei Wang, Xize Cheng, Siqi Zheng, Zhou Zhao

PyTorch Implementation of [AudioLCM]: a efficient and high-quality text-to-audio generation with latent consistency model.

<!-- [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2301.12661)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/AIGC-Audio/Make_An_Audio)
[![GitHub Stars](https://img.shields.io/github/stars/Text-to-Audio/Make-An-Audio?style=social)](https://github.com/Text-to-Audio/Make-An-Audio) -->

We provide our implementation and pretrained models as open source in this repository.

Visit our [demo page](https://audiolcm.github.io/) for audio samples.

<!-- [Text-to-Audio HuggingFace Space](https://huggingface.co/spaces/AIGC-Audio/Make_An_Audio) | [Audio Inpainting HuggingFace Space](https://huggingface.co/spaces/AIGC-Audio/Make_An_Audio_inpaint) -->

## News
<!-- - Jan, 2023: **[Make-An-Audio](https://arxiv.org/abs/2207.06389)** submitted to arxiv. -->
- June, 2024: **[AudioLCM]** released in Github. 

## Quick Started
We provide an example of how you can generate high-fidelity samples quickly using AudioLCM.

To try on your own dataset, simply clone this repo in your local machine provided with NVIDIA GPU + CUDA cuDNN and follow the below instructions.


### Dependencies
See requirements in `requirement.txt`:

## Inference with pretrained model
```bash
python scripts/txt2audio_for_lcm.py  --ddim_steps 2 -b configs/audiolcm.yaml --sample_rate 16000 --vocoder-ckpt  vocoder/logs/bigvnat16k93.5w --outdir results --test-dataset audiocaps  -r ckpt/audiolcm.ckpt
```
# Train
## dataset preparation
We can't provide the dataset download link for copyright issues. We provide the process code to generate melspec.  
Before training, we need to construct the dataset information into a tsv file, which includes name (id for each audio), dataset (which dataset the audio belongs to), audio_path (the path of .wav file),caption (the caption of the audio) ,mel_path (the processed melspec file path of each audio). We provide a tsv file of audiocaps test set: ./audiocaps_test_16000_struct.tsv as a sample.
### generate the melspec file of audio
Assume you have already got a tsv file to link each caption to its audio_path, which mean the tsv_file have "name","audio_path","dataset" and "caption" columns in it.
To get the melspec of audio, run the following command, which will save mels in ./processed
```bash
python ldm/data/preprocess/mel_spec.py --tsv_path tmp.tsv
```
Add the duration into the tsv file
```bash
python ldm/data/preprocess/add_duration.py
```
## Train variational autoencoder
Assume we have processed several datasets, and save the .tsv files in data/*.tsv . Replace **data.params.spec_dir_path** with the **data**(the directory that contain tsvs) in the config file. Then we can train VAE with the following command. If you don't have 8 gpus in your machine, you can replace --gpus 0,1,...,gpu_nums
```bash
python main.py --base configs/train/vae.yaml -t --gpus 0,1,2,3,4,5,6,7
```
The training result will be save in ./logs/
## train latent diffsuion
After Trainning VAE, replace model.params.first_stage_config.params.ckpt_path with your trained VAE checkpoint path in the config file.
Run the following command to train Diffusion model
```bash
python main.py --base configs/autoencoder1d.yaml -t  --gpus 0,1,2,3,4,5,6,7
```
The training result will be save in ./logs/
# Evaluation
## generate audiocaps samples
```bash
python scripts/txt2audio_for_lcm.py  --ddim_steps 2 -b configs/audiolcm.yaml --sample_rate 16000 --vocoder-ckpt  vocoder/logs/bigvnat16k93.5w --outdir results --test-dataset audiocaps  -r ckpt/audiolcm.ckpt
```

## calculate FD,FAD,IS,KL
install [audioldm_eval](https://github.com/haoheliu/audioldm_eval) by
```bash
git clone git@github.com:haoheliu/audioldm_eval.git
```
Then test with:
```bash
python scripts/test.py --pred_wavsdir {the directory that saves the audios you generated} --gt_wavsdir {the directory that saves audiocaps test set waves}
```
## calculate Clap_score
```bash
python wav_evaluation/cal_clap_score.py --tsv_path {the directory that saves the audios you generated}/result.tsv
```


## Acknowledgements
This implementation uses parts of the code from the following Github repos:
[Make-An-Audio](https://github.com/Text-to-Audio/Make-An-Audio)
[CLAP](https://github.com/LAION-AI/CLAP),
[Stable Diffusion](https://github.com/CompVis/stable-diffusion),
as described in our code.

<!-- ## Citations ##
If you find this code useful in your research, please consider citing:
```bibtex
@article{huang2023make,
  title={Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models},
  author={Huang, Rongjie and Huang, Jiawei and Yang, Dongchao and Ren, Yi and Liu, Luping and Li, Mingze and Ye, Zhenhui and Liu, Jinglin and Yin, Xiang and Zhao, Zhou},
  journal={arXiv preprint arXiv:2301.12661},
  year={2023}
}
``` -->

# Disclaimer ##
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.
