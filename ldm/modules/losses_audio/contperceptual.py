import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.insert(0, '.')  # nopep8
from ldm.modules.losses_audio.vqperceptual import *

def discriminator_loss_mse(disc_real_outputs, disc_generated_outputs):
    r_losses = 0
    g_losses = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        r_losses += r_loss
        g_losses += g_loss
    r_losses = r_losses / len(disc_real_outputs)
    g_losses = g_losses / len(disc_real_outputs)
    total = 0.5 * (r_losses + g_losses)
    return total

class LPAPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3,disc_hidden_size=64, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge",r1_reg_weight=5):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla","mse"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        if self.perceptual_weight > 0:
            raise RuntimeError("don't use perceptual loss")
            # self.perceptual_loss =  LPAPS().eval()# LPIPS用于日常图像，而LPAPS用于梅尔谱图
        
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 ndf = disc_hidden_size,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ).apply(weights_init) # h=8,w/(2**disc_num_layers) - 2
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == 'mse':
            self.disc_loss = discriminator_loss_mse
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"LPAPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.r1_reg_weight = r1_reg_weight


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", weights=None):
        if len(inputs.shape) == 3:
            inputs,reconstructions = inputs.unsqueeze(1),reconstructions.unsqueeze(1)            
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            # print(f"p_loss {p_loss}")
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                d_real_in = inputs.contiguous().detach()
                d_real_in.requires_grad = True
                logits_real = self.discriminator(d_real_in)
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
            
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake) # logits_real越大，logits_fake越小说明discriminator越强
            if self.r1_reg_weight > 0 and split=='train':
                r1_grads = torch.autograd.grad(outputs=[logits_real.sum()], inputs=[d_real_in], create_graph=True, only_inputs=True)
                r1_grads = r1_grads[0]
                r1_penalty = r1_grads.square().mean()  
                d_loss += self.r1_reg_weight * r1_penalty
            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            if self.r1_reg_weight and split=='train':
                log["{}/r1_prnalty".format(split)] = r1_penalty
            return d_loss, log


