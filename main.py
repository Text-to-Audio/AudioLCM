import argparse, os, sys, datetime, glob
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import time
import torch
import torch.distributed as dist
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import soundfile
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from functools import partial
import ldm
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback,LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from ldm.util import instantiate_from_config
from ldm.data.joinaudiodataset_anylen import JoinManifestSpecs
from ldm.data.joinaudiodataset_struct_sample_anylen import JoinManifestSpecs


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
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
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "-val",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="validation",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "--test-repeat",
        type=int,
        default=1,
        help="repeat each caption for t times in test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser

def getrank():
    def is_dist_avail_and_initialized():
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):# batchloader outputshape should be (b,h,w,c) and it will be permuted to (b,c,h,w) in autoencoder.get_input()
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        init_fn = None
        if isinstance(self.datasets["train"],ldm.data.joinaudiodataset_anylen.JoinManifestSpecs):
            from ldm.data.joinaudiodataset_anylen import DDPIndexBatchSampler
            dataset = self.datasets["train"]
            batch_sampler = DDPIndexBatchSampler(indices=dataset.ordered_indices(),batch_size=self.batch_size,shuffle=True,drop_last=True)
            return DataLoader(dataset, batch_sampler=batch_sampler,sampler=None,
                          num_workers=self.num_workers, collate_fn=dataset.collater,
                          worker_init_fn=init_fn)
        elif isinstance(self.datasets["train"],ldm.data.joinaudiodataset_struct_anylen.JoinManifestSpecs):
            from ldm.data.joinaudiodataset_struct_anylen import DDPIndexBatchSampler
            dataset = self.datasets["train"]
            batch_sampler = DDPIndexBatchSampler(indices=dataset.ordered_indices(),batch_size=self.batch_size,shuffle=True,drop_last=True)
            return DataLoader(dataset, batch_sampler=batch_sampler,sampler=None,
                          num_workers=self.num_workers, collate_fn=dataset.collater,
                          worker_init_fn=init_fn)
        elif isinstance(self.datasets["train"],ldm.data.joinaudiodataset_struct_sample_anylen.JoinManifestSpecs):
            from ldm.data.joinaudiodataset_struct_sample_anylen import DDPIndexBatchSampler
            dataset = self.datasets["train"]
            main_indices,other_indices = dataset.ordered_indices()
            # main_indices = dataset.ordered_indices()
            batch_sampler = DDPIndexBatchSampler(main_indices,other_indices,batch_size=self.batch_size,shuffle=True,drop_last=True)
            # batch_sampler = DDPIndexBatchSampler(main_indices,batch_size=self.batch_size,shuffle=True,drop_last=True)
            loader = DataLoader(dataset, batch_sampler=batch_sampler,sampler=None,
                          num_workers=self.num_workers, collate_fn=dataset.collater,
                          worker_init_fn=init_fn)
            print("train_loader_length",len(loader))
            return loader
        else:
            return DataLoader(self.datasets["train"], batch_size=self.batch_size ,# sampler=DistributedSampler # np.arange(100),
                            num_workers=self.num_workers, shuffle=True,
                            worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        init_fn = None
        if isinstance(self.datasets["validation"],ldm.data.joinaudiodataset_struct_anylen.JoinManifestSpecs):
            from ldm.data.joinaudiodataset_struct_anylen import DDPIndexBatchSampler
            dataset = self.datasets["validation"]
            batch_sampler = DDPIndexBatchSampler(indices=dataset.ordered_indices(),batch_size=self.batch_size,shuffle=shuffle,drop_last=True)
            return DataLoader(dataset, batch_sampler=batch_sampler,sampler=None,
                          num_workers=self.num_workers, collate_fn=dataset.collater,
                          worker_init_fn=init_fn)
        if isinstance(self.datasets["validation"],JoinManifestSpecs):
            from ldm.data.joinaudiodataset_struct_sample_anylen import DDPIndexBatchSampler
            dataset = self.datasets["validation"]
            main_indices,other_indices = dataset.ordered_indices()
            batch_sampler = DDPIndexBatchSampler(main_indices,other_indices,batch_size=self.batch_size,shuffle=shuffle,drop_last=True)
            return DataLoader(dataset, batch_sampler=batch_sampler,sampler=None,
                          num_workers=self.num_workers, collate_fn=dataset.collater,
                          worker_init_fn=init_fn)
        else:
            return DataLoader(self.datasets["validation"],
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            worker_init_fn=init_fn,
                            shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        init_fn = None
        # do not shuffle dataloader for iterable dataset
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SpectrogramDataModuleFromConfig(DataModuleFromConfig):
    '''avoiding duplication of hyper-parameters in the config by gross patching here '''
    def __init__(self, batch_size, num_workers,spec_dir_path=None,main_spec_dir_path=None,other_spec_dir_path=None,
                  mel_num=None, spec_len=None, spec_crop_len=1248,drop=0,mode='pad',
                 require_caption=True, train=None, validation=None, test=None, predict=None, wrap=False):
        specs_dataset_cfg = {
            'spec_dir_path': spec_dir_path,
            'main_spec_dir_path':main_spec_dir_path,
            'other_spec_dir_path':other_spec_dir_path,
            'require_caption': require_caption,
            'mel_num': mel_num,
            'spec_len': spec_len,
            'spec_crop_len': spec_crop_len,
            'mode': mode,
            'drop': drop
        }
        for name, split in {'train': train, 'validation': validation, 'test': test}.items():
            if split is not None:
                split.params.specs_dataset_cfg = specs_dataset_cfg
        super().__init__(batch_size, train, validation, test, predict, wrap, num_workers)



class SetupCallback(Callback):# will not load ckpt, just set directories for the experiment
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_exception(self, trainer, pl_module, exception):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, increase_log_steps=True,
                 disabled=False, log_on_batch_idx=False, log_first_step=False,melvmin=0,melvmax=1,
                 log_images_kwargs=None,**kwargs):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TensorBoardLogger: self._log,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.melvmin=melvmin
        self.melvmax=melvmax

    @rank_zero_only
    def _log(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            fig = plt.figure()
            plt.pcolor(grid.mean(dim=0),vmin=self.melvmin,vmax=self.melvmax)
            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_figure(tag, fig,global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)#  c=3,h,w
            grid = grid.mean(dim=0)# to 1 channel
            grid = grid.numpy()
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            plt.imsave(path,grid,vmin=self.melvmin,vmax=self.melvmax)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step

        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():# 这里会调用ddpm中的log_images
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)# images is a dict

            for k in images.keys():
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")
        # pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class AudioLogger(ImageLogger):
    def __init__(self, batch_frequency, max_images, increase_log_steps=True, melvmin=0,melvmax=1,disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, for_specs=False, vocoder_cfg=None, spec_dir_name=None, sample_rate=None,**kwargs):
        super().__init__(batch_frequency, max_images,  increase_log_steps,  disabled, log_on_batch_idx, log_first_step, melvmin,melvmax,log_images_kwargs)
        self.for_specs = for_specs
        self.spec_dir_name = spec_dir_name
        self.sample_rate = sample_rate
        print('We will not save audio for conditioning and conditioning_rec')
        if self.for_specs:
            self.vocoder = instantiate_from_config(vocoder_cfg)

    def _visualize_attention(self, attention, scale_by_prior=True):
        if scale_by_prior:
            B, H, T, T = attention.shape
            # attention weight is 1/T: if we have a seq with length 3 the weights are 1/3, 1/3, and 1/3
            # making T by T matrix with zeros in the upper triangular part
            attention_uniform_prior = 1 / torch.arange(1, T+1).view(1, T, 1).repeat(B, 1, T)
            attention_uniform_prior = attention_uniform_prior.tril().view(B, 1, T, T).to(attention.device)
            attention = attention - attention_uniform_prior

        attention_agg = attention.sum(dim=1, keepdims=True)
        return attention_agg

    def _log_rec_audio(self, specs, tag, global_step, pl_module=None, save_rec_path=None):

        # specs are (B, 1, F, T)
        for i, spec in enumerate(specs):
            spec = spec.data.squeeze(0).cpu().numpy()
            if spec.shape[0] != 80: continue
            wave = self.vocoder.vocode(spec)
            wave = torch.from_numpy(wave).unsqueeze(0)
            if pl_module is not None:
                pl_module.logger.experiment.add_audio(f'{tag}_{i}', wave, global_step, self.sample_rate)
            # in case we would like to save it on disk
            if save_rec_path is not None:
                soundfile.write(save_rec_path, wave.squeeze(0).numpy(), self.sample_rate, 'FLOAT')

    @rank_zero_only
    def _log(self, pl_module, images, batch_idx, split):
        for k in images: # images is a dict,images[k]'s shape is (B,C,H,W)
            tag = f'{split}/{k}'
            if self.for_specs:
                # flipping values along frequency dim, otherwise mels are upside-down (1, F, T)
                grid = torchvision.utils.make_grid(images[k].flip(dims=(2,)), nrow=1)
                # also reconstruct waveform given the spec and inv_transform
                if k not in ['conditioning', 'conditioning_rec', 'att_nopix', 'att_half', 'att_det']:
                    self._log_rec_audio(images[k], tag, pl_module.global_step, pl_module=pl_module)
            else:
                grid = torchvision.utils.make_grid(images[k])# (B,C=1 or 3,H,W) -> (C=3,B*H,W)
                # attention is already in [0, 1] therefore ignoring this line
            fig = plt.figure()
            plt.pcolor(grid.mean(dim=0),vmin=self.melvmin,vmax=self.melvmax)
            pl_module.logger.experiment.add_figure(tag, fig,global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = grid.mean(dim=0)
            grid = grid.numpy()
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            plt.imsave(path,grid,vmin=self.melvmin,vmax=self.melvmax)

            # also save audio on the disk
            if self.for_specs:
                tag = f'{split}/{k}'
                filename = filename.replace('.png', '.wav')
                path = os.path.join(root, filename)
                if k not in ['conditioning', 'conditioning_rec', 'att_nopix', 'att_half', 'att_det']:
                    self._log_rec_audio(images[k], tag, global_step, save_rec_path=path)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):# ,outputs： outputs positional argument has been removed in the later pytorch-lighning version。
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.ckpt_path = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        print(f"opt.base:{opt.base}")
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["strategy"] = "ddp" # "ddp" # "ddp_find_unused_parameters_false"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)

        if not "gpus" in trainer_config:
            del trainer_config["strategy"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "name": "tensorboard",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["tensorboard"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)


        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
                "save_top_k": 5,
            }
        }
        # use valitdation monitor:
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")


        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 5000,
                    "max_images": 4,
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }

        # patching the default config for the spectrogram input
        # if 'Spectrogram' in config.data.target:
        #    spec_dir_name = Path(config.data.params.spec_dir_path).name
        #    default_callbacks_cfg['image_logger']['params']['spec_dir_name'] = spec_dir_name
        #    default_callbacks_cfg['image_logger']['params']['sample_rate'] = config.data.params.sample_rate

        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 10000,
                         'save_weights_only': True
                     }
                    }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'ckpt_path'):# false for the former
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.ckpt_path
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]



        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  

        ##### data #####
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)
        print(f"#####  trainer.logdir:{trainer.logdir}  #####")
        # run
        if opt.train:
            try:
                if hasattr(opt,'ckpt_path'):
                    trainer.fit(model, data,ckpt_path = opt.ckpt_path)
                else:
                    trainer.fit(model, data)
            except Exception:
                melk()
                raise
        elif opt.val:
            trainer.validate(model, data)
        if not opt.no_test and not trainer.interrupted:
            if not opt.train and hasattr(opt,'ckpt_path'):# just test the ckeckpoint, without training
                trainer.test(model, data, ckpt_path = opt.ckpt_path)
            else:# test the model after trainning
                trainer.test(model, data)               
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
