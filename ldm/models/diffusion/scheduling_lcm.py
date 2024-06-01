"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import List, Optional, Tuple, Union
from ldm.util import randn_tensor
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class LCMSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.original_inference_steps = 100
        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, self.ddpm_num_timesteps)[::-1].copy().astype(np.int64))
        self.custom_timesteps = False
        self.timestep_scaling = 10.0
        self.prediction_type = 'epsilon'


    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_discretize="uniform", verbose=True):
        # self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
        #                                           num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        # alphas_cumprod = self.model.alphas_cumprod
        # assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        # beta_start = 0.00085
        # beta_end = 0.012
        # self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, self.ddpm_num_timesteps, dtype=torch.float32) ** 2
        # self.alphas = 1.0 - self.betas
        # self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)
        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # # calculations for diffusion q(x_t | x_{t-1}) and others
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        # self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        # self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        # self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        

        # # ddim sampling parameters
        # ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
        #                                                                            ddim_timesteps=self.ddim_timesteps,
        #                                                                            eta=ddim_eta,verbose=verbose)
        # self.register_buffer('ddim_sigmas', ddim_sigmas)
        # self.register_buffer('ddim_alphas', ddim_alphas)
        # self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        # self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        # sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
        #     (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
        #                 1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        # self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")
        
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def step_index(self):
        return self._step_index

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        original_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        strength: int = 1.0,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            original_inference_steps (`int`, *optional*):
                The original number of inference steps, which will be used to generate a linearly-spaced timestep
                schedule (which is different from the standard `diffusers` implementation). We will then take
                `num_inference_steps` timesteps from this schedule, evenly spaced in terms of indices, and use that as
                our final timestep schedule. If not set, this will default to the `original_inference_steps` attribute.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps on the training/distillation timestep
                schedule is used. If `timesteps` is passed, `num_inference_steps` must be `None`.
        """
        # 0. Check inputs
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Must pass exactly one of `num_inference_steps` or `custom_timesteps`.")

        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")

        # 1. Calculate the LCM original training/distillation timestep schedule.
        original_steps = (
            original_inference_steps if original_inference_steps is not None else self.original_inference_steps
        )

        if original_steps > self.ddpm_num_timesteps:
            raise ValueError(
                f"`original_steps`: {original_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.ddpm_num_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.ddpm_num_timesteps} timesteps."
            )
        # import ipdb
        # ipdb.set_trace()
        # LCM Timesteps Setting
        # The skipping step parameter k from the paper.
        k = self.ddpm_num_timesteps // original_steps
        # LCM Training/Distillation Steps Schedule
        # Currently, only a linearly-spaced schedule is supported (same as in the LCM distillation scripts).
        lcm_origin_timesteps = np.asarray(list(range(1, int(original_steps * strength) + 1))) * k - 1

        # 2. Calculate the LCM inference timestep schedule.
        if timesteps is not None:
            # 2.1 Handle custom timestep schedules.
            train_timesteps = set(lcm_origin_timesteps)
            non_train_timesteps = []
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`custom_timesteps` must be in descending order.")

                if timesteps[i] not in train_timesteps:
                    non_train_timesteps.append(timesteps[i])

            if timesteps[0] >= self.ddpm_num_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.ddpm_num_timesteps}."
                )

            # Raise warning if timestep schedule does not start with self.config.num_train_timesteps - 1
            if strength == 1.0 and timesteps[0] != self.ddpm_num_timesteps - 1:
                logger.warning(
                    f"The first timestep on the custom timestep schedule is {timesteps[0]}, not"
                    f" `self.ddpm_num_timesteps - 1`: {self.ddpm_num_timesteps - 1}. You may get"
                    f" unexpected results when using this timestep schedule."
                )

            # Raise warning if custom timestep schedule contains timesteps not on original timestep schedule
            if non_train_timesteps:
                logger.warning(
                    f"The custom timestep schedule contains the following timesteps which are not on the original"
                    f" training/distillation timestep schedule: {non_train_timesteps}. You may get unexpected results"
                    f" when using this timestep schedule."
                )

            # Raise warning if custom timestep schedule is longer than original_steps
            if len(timesteps) > original_steps:
                logger.warning(
                    f"The number of timesteps in the custom timestep schedule is {len(timesteps)}, which exceeds the"
                    f" the length of the timestep schedule used for training: {original_steps}. You may get some"
                    f" unexpected results when using this timestep schedule."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.num_inference_steps = len(timesteps)
            self.custom_timesteps = True

            # Apply strength (e.g. for img2img pipelines) (see StableDiffusionImg2ImgPipeline.get_timesteps)
            init_timestep = min(int(self.num_inference_steps * strength), self.num_inference_steps)
            t_start = max(self.num_inference_steps - init_timestep, 0)
            timesteps = timesteps[t_start * self.order :]
            # TODO: also reset self.num_inference_steps?
        else:
            # 2.2 Create the "standard" LCM inference timestep schedule.
            if num_inference_steps > self.ddpm_num_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.ddpm_num_timesteps`:"
                    f" {self.ddpm_num_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.ddpm_num_timesteps} timesteps."
                )

            skipping_step = len(lcm_origin_timesteps) // num_inference_steps

            if skipping_step < 1:
                raise ValueError(
                    f"The combination of `original_steps x strength`: {original_steps} x {strength} is smaller than `num_inference_steps`: {num_inference_steps}. Make sure to either reduce `num_inference_steps` to a value smaller than {int(original_steps * strength)} or increase `strength` to a value higher than {float(num_inference_steps / original_steps)}."
                )

            self.num_inference_steps = num_inference_steps

            if num_inference_steps > original_steps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `original_inference_steps`:"
                    f" {original_steps} because the final timestep schedule will be a subset of the"
                    f" `original_inference_steps`-sized initial timestep schedule."
                )

            # LCM Inference Steps Schedule
            lcm_origin_timesteps = lcm_origin_timesteps[::-1].copy()
            # Select (approximately) evenly spaced indices from lcm_origin_timesteps.
            inference_indices = np.linspace(0, len(lcm_origin_timesteps), num=num_inference_steps, endpoint=False)
            inference_indices = np.floor(inference_indices).astype(np.int64)
            timesteps = lcm_origin_timesteps[inference_indices]

        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.long)

        self._step_index = None

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
    def retrieve_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            scheduler (`SchedulerMixin`):
                The scheduler to get timesteps from.
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                    Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                    timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                    must be `None`.

        Returns:
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """
        if timesteps is not None:
            self.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = self.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = self.timesteps
        return timesteps, num_inference_steps   

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               verbose=True,
               x_T=None,
               guidance_scale=5.,
               original_inference_steps=50,
               timesteps=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(verbose=verbose)
        self.num_inference_steps = S
        # sampling
        if len(shape)==3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        else:
            C, T = shape
            size = (batch_size, C, T) 

        samples, intermediates = self.lcm_sampling(conditioning, size,
                                                    x_T=x_T,
                                                    guidance_scale=guidance_scale,
                                                    original_inference_steps=original_inference_steps,
                                                    timesteps=timesteps
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def lcm_sampling(self, cond, shape,
                      x_T=None,
                      guidance_scale=1.,original_inference_steps=100,timesteps=None):
        device = self.model.betas.device
        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.num_inference_steps, device, timesteps, original_inference_steps=original_inference_steps
        )
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        
        w = torch.tensor(guidance_scale - 1).repeat(b)
        w_embedding = self.get_guidance_scale_embedding(w, embedding_dim=256).to(
            device=device, dtype=img.dtype
        )
        
        # import ipdb
        # ipdb.set_trace()
        # 8. LCM MultiStep Sampling Loop:
        num_warmup_steps = len(timesteps) - num_inference_steps
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                img = img.to(cond.dtype)
                ts = torch.full((b,), t, device=device, dtype=torch.long)
                # model prediction (v-prediction, eps, x)
                model_pred = self.model.apply_model(img, ts, cond,self.model.unet, w_cond=w_embedding)

                # compute the previous noisy sample x_t -> x_t-1
                img, denoised = self.step(model_pred, t, img, return_dict=False)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps):
                    progress_bar.update()
        return denoised, img

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index
    def _init_step_index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        index_candidates = (self.timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        if len(index_candidates) > 1:
            step_index = index_candidates[1]
        else:
            step_index = index_candidates[0]

        self._step_index = step_index.item()

    def get_scalings_for_boundary_condition_discrete(self, timestep):
        self.sigma_data = 0.5  # Default: 0.5
        scaled_timestep = timestep * self.timestep_scaling

        c_skip = self.sigma_data**2 / (scaled_timestep**2 + self.sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out

    @torch.no_grad()
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] or `tuple`.
        Returns:
            [`~schedulers.scheduling_utils.LCMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        
        if self.step_index is None:
            self._init_step_index(timestep)
        # 1. get previous step value
        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = timestep

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 3. Get scalings for boundary conditions

        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        # 4. Compute the predicted original sample x_0 based on the model parameterization
        if self.prediction_type == "epsilon":  # noise-prediction
            predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.prediction_type == "sample":  # x-prediction
            predicted_original_sample = model_output
        elif self.prediction_type == "v_prediction":  # v-prediction
            predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction` for `LCMScheduler`."
            )


        # 5. Denoise model output using boundary conditions
        denoised = c_out * predicted_original_sample + c_skip * sample

        # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        # Noise is not used on the final timestep of the timestep schedule.
        # This also means that noise is not used for one-step sampling.
        if self.step_index != self.num_inference_steps - 1:
            noise = torch.randn(model_output.shape, device=model_output.device)
            prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
        else:
            prev_sample = denoised

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample, denoised)

        return prev_sample, denoised