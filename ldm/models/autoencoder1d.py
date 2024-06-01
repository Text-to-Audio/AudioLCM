"""
与autoencoder.py的区别在于，autoencoder.py是(B,1,80,T) ->(B,C,80/8,T/8),现在vae要变成(B,80,T) -> (B,80/downsample_c,T/downsample_t)
"""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
from packaging import version
import numpy as np
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from torch.optim.lr_scheduler import LambdaLR
from ldm.util import instantiate_from_config


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 embed_dim,
                 ddconfig,
                 lossconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder1D(**ddconfig)
        self.decoder = Decoder1D(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv1d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv1d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"AutoencoderKL Restored from {path} Done")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        assert len(x.shape) == 3
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        # print(inputs.shape)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def test_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)# inputs shape:(b,mel_len,T)
        reconstructions, posterior = self(inputs)# reconstructions:(b,mel_len,T)
        mse_loss = torch.nn.functional.mse_loss(reconstructions,inputs)
        self.log('test/mse_loss',mse_loss)
          
        test_ckpt_path = os.path.basename(self.trainer.tested_ckpt_path)
        savedir = os.path.join(self.trainer.log_dir,f'output_imgs_{test_ckpt_path}','fake_class')
        if batch_idx == 0:
            print(f"save_path is: {savedir}")
        if not os.path.exists(savedir):
            os.makedirs(savedir)
            print(f"save_path is: {savedir}")

        file_names = batch['f_name']
        # print(f"reconstructions.shape:{reconstructions.shape}",file_names)
        # reconstructions = (reconstructions + 1)/2 # to mel scale  
        reconstructions = reconstructions.cpu().numpy() # squuze channel dim
        for b in range(reconstructions.shape[0]):
            vname_num_split_index = file_names[b].rfind('_')# file_names[b]:video_name+'_'+num
            v_n,num = file_names[b][:vname_num_split_index],file_names[b][vname_num_split_index+1:]
            save_img_path = os.path.join(savedir, f'{v_n}.npy') # f'{v_n}_sample_{num}.npy'   f'{v_n}.npy'
            np.save(save_img_path,reconstructions[b])
        
        return None
        
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        
        if not only_inputs:
            xrec, posterior = self(x)
            log["samples"] = self.decode(torch.randn_like(posterior.sample())).unsqueeze(1) # (b,1,H,W)
            log["reconstructions"] = xrec.unsqueeze(1)
        log["inputs"] = x.unsqueeze(1)
        return log


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class ResnetBlock1D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512,kernel_size = 3):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=kernel_size//2)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=kernel_size//2)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=kernel_size,
                                                     stride=1,
                                                     padding=kernel_size//2)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class AttnBlock1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,t,c = q.shape
        q = q.permute(0,2,1)   # b,t,c   
        w_ = torch.bmm(q,k)     # b,t,t   w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # if still 2d attn (q:b,hw,c ,k:b,c,hw -> w_:b,hw,hw)
        w_ = w_ * (int(t)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        w_ = w_.permute(0,2,1)   # b,t,t (first t of k, second of q)
        h_ = torch.bmm(v,w_)     # b,c,t (t of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        h_ = self.proj_out(h_)

        return x+h_

class Upsample1D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest") # support 3D tensor(B,C,T)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample1D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        return x

class Encoder1D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_layers = [],down_layers = [], dropout=0.0, resamp_with_conv=True, in_channels,
                 z_channels, double_z=True,kernel_size=3, **ignore_kwargs):
        """ out_ch is only used in decoder,not used here
        """
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        print(f"downsample rates is {2**len(down_layers)}")
        self.down_layers = down_layers
        self.attn_layers = attn_layers
        self.conv_in = torch.nn.Conv1d(in_channels,
                                       self.ch,
                                       kernel_size=kernel_size,
                                       stride=1,
                                       padding=kernel_size//2)

        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        # downsampling
        self.down = nn.ModuleList()
        for i_level in range(self.num_layers):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock1D(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         kernel_size=kernel_size))
                block_in = block_out
                if i_level in attn_layers:
                    # print(f"add attn in layer:{i_level}")
                    attn.append(AttnBlock1D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level in down_layers:
                down.downsample = Downsample1D(block_in, resamp_with_conv)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       kernel_size=kernel_size)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       kernel_size=kernel_size)

        # end
        self.norm_out = Normalize(block_in)# GroupNorm
        self.conv_out = torch.nn.Conv1d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=kernel_size//2)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_layers):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level in self.down_layers:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
    
class Decoder1D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_layers = [],down_layers = [], dropout=0.0,kernel_size=3, resamp_with_conv=True, in_channels,
                z_channels, give_pre_end=False, tanh_out=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.down_layers = [i+1 for i in down_layers] # each downlayer add one
        print(f"upsample rates is {2**len(down_layers)}")
        
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_layers-1]


        # z to block_in
        self.conv_in = torch.nn.Conv1d(z_channels,
                                       block_in,
                                       kernel_size=kernel_size,
                                       stride=1,
                                       padding=kernel_size//2)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_layers)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock1D(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if i_level in attn_layers:
                    # print(f"add attn in layer:{i_level}")
                    attn.append(AttnBlock1D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level in self.down_layers:
                up.upsample = Upsample1D(block_in, resamp_with_conv)
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=kernel_size//2)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_layers)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level in self.down_layers:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h