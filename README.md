#### Huadai Liu, Rongjie Huang, Yang Liu, Hengyuan Cao, Jialei Wang, Xize Cheng, Siqi Zheng, Zhou Zhao

PyTorch Implementation of [AudioLCM]: an efficient and high-quality text-to-audio generation with latent consistency model.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2406.00356v1)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/AIGC-Audio/AudioLCM)
[![GitHub Stars](https://img.shields.io/github/stars/liuhuadai/AudioLCM?style=social)](https://github.com/liuhuadai/AudioLCM)

We provide our implementation and pretrained models as open-source in this repository.

Visit our [demo page](https://audiolcm.github.io/) for audio samples.

	@@ -53,16 +53,17 @@ useful_ckpts/
### Dependencies
See requirements in `requirement.txt`:

## Inference with a pre-trained model
```bash
python scripts/txt2audio_for_lcm.py  --ddim_steps 2 -b configs/audiolcm.yaml --sample_rate 16000 --vocoder-ckpt  vocoder/logs/bigvnat16k93.5w --outdir results --test-dataset audiocaps  -r ckpt/audiolcm.ckpt
```

## Dataset preparation
- We can't provide the dataset download link for copyright issues. We provide the process code to generate melspec.  
- Before training, we need to construct the dataset information into a tsv file, which includes the name (id for each audio), dataset (which dataset the audio belongs to), audio_path (the path of .wav file),caption (the caption of the audio) ,mel_path (the processed melspec file path of each audio). 
- We provide a tsv file of the audiocaps test set: ./audiocaps_test_16000_struct.tsv as a sample.
### Generate the melspec file of audio
Assume you have already got a tsv file to link each caption to its audio_path, which means the tsv_file has "name","audio_path","dataset" and "caption" columns in it.
To get the melspec of audio, run the following command, which will save mels in ./processed
```bash
python ldm/data/preprocess/mel_spec.py --tsv_path tmp.tsv
	@@ -76,34 +77,16 @@ Assume we have processed several datasets, and save the .tsv files in data/*.tsv
```bash
python main.py --base configs/train/vae.yaml -t --gpus 0,1,2,3,4,5,6,7
```
The training result will be saved in ./logs/
## Train latent diffsuion
After Training VAE, replace model.params.first_stage_config.params.ckpt_path with your trained VAE checkpoint path in the config file.
Run the following command to train the Diffusion model
```bash
python main.py --base configs/autoencoder1d.yaml -t  --gpus 0,1,2,3,4,5,6,7
```
The training result will be saved in ./logs/
## Evaluation
Please refer to [Make-An-Audio](https://github.com/Text-to-Audio/Make-An-Audio?tab=readme-ov-file#evaluation)
