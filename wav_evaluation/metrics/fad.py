"""
Calculate Frechet Audio Distance betweeen two audio directories.

Frechet distance implementation adapted from: https://github.com/mseitzer/pytorch-fid

VGGish adapted from: https://github.com/harritaylor/torchvggish
"""
import os
import numpy as np
from glob import glob
import torch
from torch import nn
from scipy import linalg
from tqdm import tqdm
import soundfile as sf
import resampy
from multiprocessing.dummy import Pool as ThreadPool
from src.torchvggish.torchvggish.vggish import  VGGishlocal
SAMPLE_RATE = 16000 # resample audio file to SAMPLE_RATE. since uses the pretrained vggish model which takes wav_data as input

def load_audio_task(fname):# load wav file and resample to SAMPLE_RATE
    wav_data, sr = sf.read(fname, dtype='int16')
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    # Convert to mono
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)

    if sr != SAMPLE_RATE:
        wav_data = resampy.resample(wav_data, sr, SAMPLE_RATE)

    return wav_data, SAMPLE_RATE

# use pretrained torchvggish as embedding extractor, and calculate the statistic of wav file
class FrechetAudioDistance:
    def __init__(self, use_pca=False, use_activation=False, verbose=False, audio_load_worker=8):
        # self.__get_model(use_pca=use_pca, use_activation=use_activation)
        self.__get_local_model(local_path='src/torchvggish/docs',use_pca=use_pca, use_activation=use_activation)
        self.verbose = verbose
        self.audio_load_worker = audio_load_worker
    
    def __get_model(self, use_pca=False, use_activation=False):
        """
        Params:
        -- x   : Either 
            (i) a string which is the directory of a set of audio files, or
            (ii) a np.ndarray of shape (num_samples, sample_length)
        """
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        if not use_pca:
            self.model.postprocess = False
        if not use_activation:
            self.model.embeddings = nn.Sequential(*list(self.model.embeddings.children())[:-1])
        self.model.eval()
    
    def __get_local_model(self,local_path,use_pca=False, use_activation=False):
        self.model = VGGishlocal(local_path)
        if not use_pca:
            self.model.postprocess = False
        if not use_activation:
            self.model.embeddings = nn.Sequential(*list(self.model.embeddings.children())[:-1])
        self.model.eval()

    def get_embeddings(self, x, sr=16000):
        """
        Get embeddings using VGGish model.
        Params:
        -- x    : Either 
            (i) a string which is the directory of a set of audio files, or
            (ii) a list of np.ndarray audio samples
        -- sr   : Sampling rate, if x is a list of audio samples. Default value is 16000.
        """
        embd_lst = []
        if isinstance(x, list):# np.ndarray
            try:
                for audio, sr in tqdm(x, disable=(not self.verbose)):
                    embd = self.model.forward(audio, sr)
                    if self.model.device == torch.device('cuda'):
                        embd = embd.cpu()
                    embd = embd.detach().numpy()
                    embd_lst.append(embd)
            except Exception as e:
                print("[Frechet Audio Distance] get_embeddings throw an exception: {}".format(str(e)))
        elif isinstance(x, str):
            try:
                for fname in tqdm(os.listdir(x), disable=(not self.verbose)):
                    embd = self.model.forward(os.path.join(x, fname))
                    if self.model.device == torch.device('cuda'):
                        embd = embd.cpu()
                    embd = embd.detach().numpy()
                    embd_lst.append(embd)
            except Exception as e:
                print("[Frechet Audio Distance] get_embeddings throw an exception: {}".format(str(e)))
        else:
            raise AttributeError
        # print("embd_lst_len",len(embd_lst))
        return np.concatenate(embd_lst, axis=0)
    
    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
        
        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """
        # print(f"mu1.shape:{mu1.shape},mu2.shape:{sigma1.shape}")
        mu1 = np.atleast_1d(mu1) # shape(128,)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)# shape(128,128)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        print(f"diff^2:{diff.dot(diff)}, sigma1:{np.trace(sigma1)},sigma2:{np.trace(sigma2)},2 * tr_covmean{2 * tr_covmean}")
        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)
    
    def load_audio_files(self, dir):# load_audio_task会resample
        task_results = []

        all_wav_files = glob(os.path.join(dir,"*.wav"))
        pool = ThreadPool(self.audio_load_worker)
        pbar = tqdm(total=len(all_wav_files), disable=(not self.verbose))

        def update(*a):
            pbar.update()

        if self.verbose:
            print("[Frechet Audio Distance] Loading audio from {}...".format(dir))
        for fname in all_wav_files:
            res = pool.apply_async(load_audio_task, args=(fname,), callback=update)# load_audio_task会resample
            task_results.append(res)
        pool.close()
        pool.join()     
        
        return [k.get() for k in task_results] # get return value,each is (wav_data, sample_rate)

    def score(self, background_dir, eval_dir, store_embds=False):
        try:
            audio_background = self.load_audio_files(background_dir)
            audio_eval = self.load_audio_files(eval_dir)
            print("audios len",len(audio_background),len(audio_eval))
            embds_background = self.get_embeddings(audio_background) # (N,128)
            embds_eval = self.get_embeddings(audio_eval) # (M,128)
            # print(embds_background.shape,embds_eval.shape)
            if store_embds:
                np.save("embds_background.npy", embds_background)
                np.save("embds_eval.npy", embds_eval)

            if len(embds_background) == 0:
                print("[Frechet Audio Distance] background set dir is empty, exitting...")
                return -1
            if len(embds_eval) == 0:
                print("[Frechet Audio Distance] eval set dir is empty, exitting...")
                return -1
            
            mu_background, sigma_background = self.calculate_embd_statistics(embds_background)
            mu_eval, sigma_eval = self.calculate_embd_statistics(embds_eval)

            fad_score = self.calculate_frechet_distance(
                mu_background, 
                sigma_background, 
                mu_eval, 
                sigma_eval
            )

            return fad_score
            
        except Exception as e:
            print("[Frechet Audio Distance] exception thrown, {}".format(str(e)))
            return -1

