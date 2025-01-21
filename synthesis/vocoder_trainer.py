import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
import math
import random
import numpy as np
from .hificar_dur import HiFiGANGeneratorDur
from .discriminator import HiFiGANMultiScaleMultiPeriodDiscriminator
from .losses import GeneratorAdversarialLoss, DiscriminatorAdversarialLoss, \\
                    MelSpectrogramLoss, FeatureMatchLoss, DurLoss

# TODO
# Loading pre-trained hifigan and hubert centroids
# Export function

class VocoderTrainer(LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.generator = HiFiGANGeneratorDur(**config.generator_params)
        self.discriminator = discriminator_params(**config.discriminator_params)
        self.gen_adv_loss = GeneratorAdversarialLoss(**config.loss_configs["generator_adv_loss_configs"])
        self.disc_adv_loss = DiscriminatorAdversarialLoss(**config.loss_configs["discriminator_adv_loss_configs"])
        self.mel_loss = MelSpectrogramLoss(**config.loss_configs["mel_loss_configs"])
        self.feat_match_loss = FeatureMatchLoss(**config.loss_configs["feature_match_loss_configs"])
        self.dur_loss = DurLoss(**config.loss_configs["dur_loss_configs"])
        self.automatic_optimization = False
    
    def forward_gen(self, units, ar, wav, dur, **kwargs):
        #######################
        #      Generator      #
        #######################
        wav_gen, dur_pred = self.generator(units, ar, dur)
        wav = wav.unsqueeze(1)
        mel_loss = self.mel_loss(wav_gen, wav)
        gen_disc_pred = self.discriminator(wav_gen)
        gen_adv_loss = self.gen_adv_loss(gen_disc_pred)
        with torch.no_grad():
            gt_disc_pred = self.discriminator(wav)
        feat_match_loss = self.feat_match_loss(gen_disc_pred, gt_disc_pred)
        dur_loss = self.dur_loss(dur_pred, dur)
        return {"mel_loss":mel_loss, "gen_adv_loss": gen_adv_loss, "feat_match_loss":feat_match_loss, 
               "duration_loss":dur_loss}

    def forward_disc(self, units, ar, wav, dur, **kwargs):
        #######################
        #    Discriminator    #
        #######################
        with torch.no_grad():
            wav_gen, _ = self.generator(units, ar, dur)
        wav = wav.unsqueeze(1)
        gen_disc_pred = self.discriminator(wav_gen)
        gt_disc_pred = self.discriminator(wav)
        real_loss, fake_loss = self.disc_adv_loss(gen_disc_pred, gt_disc_pred)
        disc_adv_loss = real_loss + fake_loss
        
        return {"disc_adv_loss": disc_adv_loss}
    
    def training_step(self, batch):

        optimizer_g, optimizer_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()

        self.toggle_optimizer(optimizer_d)
        disc_loss_val = 0
        outputs = self.forward_disc(**batch)
        for coef_name, coef_value in self.config.disc_loss_coef.items():
            if coef_name in outputs.keys() and coef_value>0:
                disc_loss_val += coef_value * outputs[coef_name]
                self.log(f'train_{coef_name}', outputs[coef_name])
        self.log(f'train_disc_loss', disc_loss_val)
        self.manual_backward(disc_loss_val)
        optimizer_d.step()
        optimizer_d.zero_grad()
        sch_d.step()
        self.untoggle_optimizer(optimizer_d)
        
        self.toggle_optimizer(optimizer_g)
        gen_loss_val = 0
        outputs = self.net.forward_gen(**batch)
        for coef_name, coef_value in self.config.gen_loss_coef.items():
            if coef_name in outputs.keys() and coef_value>0:
                gen_loss_val += coef_value * outputs[coef_name]
                self.log(f'train_{coef_name}', outputs[coef_name])
        self.log(f'train_gen_loss', gen_loss_val )
        self.manual_backward(gen_loss_val)
        optimizer_g.step()
        optimizer_g.zero_grad()
        sch_g.step()
        self.untoggle_optimizer(optimizer_g)
        
        return gen_loss_val

            
    def validation_step(self, batch):
        outputs = self.net.forward_gen(**batch)
        loss_val = 0
        
        for coef_name, coef_value in self.config.gen_loss_coef.items():
            if coef_name in outputs.keys():
                loss_val += coef_value * outputs[coef_name]
                self.log(f'val_{coef_name}', outputs[coef_name],sync_dist=True)
        self.log(f'val_gen_loss', loss_val, sync_dist=True)
        return loss_val
    
    
    def configure_optimizers(self):

        opt_g = torch.optim.Adam([{'params': self.net.generator.parameters()},
                                  {'params': self.net.input_layer.parameters()},
                                  {'params': self.net.spkemb_layer.parameters()},
                                  {'params': self.net.progress_encoding.parameters()}],
                                 **self.config.gen_lr_configs["optimizer"])
        opt_d = torch.optim.Adam(self.net.discriminator.parameters(), **self.config.disc_lr_configs["optimizer"])
        sch_g = torch.optim.lr_scheduler.MultiStepLR(optimizer=opt_g, **self.config.gen_lr_configs["scheduler"])
        sch_d = torch.optim.lr_scheduler.MultiStepLR(optimizer=opt_d, **self.config.disc_lr_configs["scheduler"])

        return [opt_g, opt_d], [{"scheduler": sch_g, "interval": "step"}, {"scheduler": sch_d, "interval": "step"}]
    