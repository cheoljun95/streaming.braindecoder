import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import math
import random
from torchaudio.models.rnnt import emformer_rnnt_model, RNNT
from torchaudio.transforms import RNNTLoss
from torchaudio.models.rnnt_decoder import RNNTBeamSearch
from .modules import build_predictor, build_joiner, build_transcriber, build_feature_extractor
from .tokenizer import BPETokenizer, HuBERTTokenizer, PhonemeTokenizer, GraphemeTokenizer, HuBERTTokenizerPW,SDHuBERTTokenizer
from utils.utils import anneal_coef
from utils.evaluate import get_errors
import numpy as np


class unitRNNT(nn.Module):
    """
    """
    def __init__(self, transcriber_configs, joiner_configs, predictor_configs, feature_extractor_configs,
                tokenizer_type='BPE',tokenizer_confis=None,beam_width=10, tokenizer_configs=None, step_max_tokens=100,
                use_halfprecision=True, use_tf=False, **kwargs):
        
        super().__init__()
        
        self.predictor = build_predictor(**predictor_configs)
        self.joiner = build_joiner(**joiner_configs)
        self.transcriber = build_transcriber(**transcriber_configs)
        
        self.use_halfprecision = use_halfprecision
        
        if self.use_halfprecision: 
            self.predictor = self.predictor.half()
            self.joiner = self.joiner.half()
            self.transcriber = self.transcriber.half()
            
        self.rnnt = RNNT(self.transcriber, 
                         self.predictor, 
                         self.joiner)
        
        self.blank_id=predictor_configs['num_symbols']-1
        if tokenizer_type == 'BPE':
            self.tokenizer = BPETokenizer(self.blank_id)
        elif tokenizer_type == 'HB':
            self.tokenizer = HuBERTTokenizer(**tokenizer_configs)
        elif tokenizer_type == 'HBpw':
            self.tokenizer = HuBERTTokenizerPW(**tokenizer_configs)
        elif tokenizer_type == 'PH':
            self.tokenizer = PhonemeTokenizer(**tokenizer_configs)
        elif tokenizer_type == 'GR':
            self.tokenizer = GraphemeTokenizer(**tokenizer_configs)
        elif tokenizer_type == 'SDHB':
            self.tokenizer = SDHuBERTTokenizer(**tokenizer_configs)
        else:
            raise NotImplemented
        
        self.feature_extractor = build_feature_extractor(**feature_extractor_configs) 
        self.loss=RNNTLoss(reduction='sum')
        if self.use_halfprecision: 
            self.loss = self.loss.half()
        
        self.token_decoder = self.tokenizer.get_decoder()
        self.remove_tokens = self.tokenizer.get_remove_tokens()
        self.search = RNNTBeamSearch(self.rnnt, self.blank_id,step_max_tokens=step_max_tokens)
        self.beam_width = beam_width
        self.use_tf = use_tf
        
    
    def forward(self, wavs=None, wav_lens=None, texts=None, do_pred=False, predictor_state=None, ecogs=None, ecog_lens=None,  **kwargs):
        """
        """
        if wavs is None:
            wavs = ecogs
            wav_lens = ecog_lens
        with torch.no_grad():
            
            targets, target_lengths, texts = self.tokenizer(texts=texts, wavs=wavs)
            if self.feature_extractor is not None:
                sources, source_lengths = self.feature_extractor(wavs, wav_lens)
            else:
                sources, source_lengths = wavs, wav_lens
        #import pdb
        #pdb.set_trace()
        targets = targets.to(sources.device).to(torch.int32)
        target_legnths = target_lengths.to(sources.device)
        source_lengths = source_lengths.to(sources.device)
        if self.use_halfprecision:
            sources=sources.half()
        source_encodings, source_lengths = self.transcriber(
            input=sources,
            lengths=source_lengths,
        )
        target_encodings, target_lengths, predictor_state = self.predictor(
            input=targets,
            lengths=target_lengths,
            state=predictor_state,
        )
        output, source_lengths, target_lengths = self.joiner(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
        )
        target_lengths = target_lengths.to(sources.device).to(torch.int32)
        source_lengths = source_lengths.to(torch.int32).clip(0,output.shape[1])
        
        
        
        loss =self.loss(logits=output,targets=targets[:,1:].contiguous(),
                        logit_lengths=source_lengths,
                        target_lengths=target_lengths-1,
                       ).mean() 
        
        if do_pred:
            transcribed_outputs=source_encodings
            with torch.no_grad():
                decoded=[self.search._search(trans[None,:l], None, beam_width=self.beam_width)[0][0] for trans,l in zip(transcribed_outputs, source_lengths)]
            pred = [self.token_decoder([t for t in dec if t not in self.remove_tokens+[self.blank_id]])
                         for dec in decoded]
        else:
            pred = []
            
        outputs = {'rnnt_loss':loss,
                   'logits':output,
                   'logit_lengths':source_lengths,
                   'targets':targets,
                   'target_lengths':target_lengths,
                   'pred':pred,
                   'texts':texts,
                   }

        return outputs
    
    
class SpeechModel(LightningModule):

    def __init__(self, loss_coef_instructions, non_val_loss=[], lr=0.001,use_cosine_lr=True, T_max=500000, **model_configs):
        super().__init__()

        self.loss_coef_dict = loss_coef_instructions
        self.lr = lr
        self.net = unitRNNT(**model_configs)
        self.non_val_loss = non_val_loss
        self.T_max = T_max
        self.use_cosine_lr = use_cosine_lr
        self.test_results = None
            
    def forward(self, **kwargs):
        return self.net(**kwargs)

    def _update_coef(self):
        for coef_name, coef_inst in self.loss_coef_dict.items():
            if 'anneal_step_range' in coef_inst.keys() and coef_inst['anneal_step_range'] is not None:
                coef_inst['value'] = anneal_coef(self.global_step, coef_inst['anneal_step_range'],
                                                 coef_inst['anneal_value_range'])
                self.log(f'{coef_name}_coef', coef_inst['value'])
    
        
        
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self._update_coef()
        outputs = self.net(**batch)
        loss_val = 0
        for coef_name, coef_inst in self.loss_coef_dict.items():
            if coef_name in outputs.keys() and coef_inst['value']>0:
                loss_val += coef_inst['value'] * outputs[coef_name]
                self.log(f'train_{coef_name}', outputs[coef_name])
        self.log(f'train_loss', loss_val)
        return loss_val

            
            
    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        outputs = self.net(**batch,do_pred=True)
        loss_val = 0
        
        for coef_name, coef_inst in self.loss_coef_dict.items():
            if coef_name in outputs.keys():
                if coef_name not in self.non_val_loss:
                    loss_val += coef_inst['value'] * outputs[coef_name]
                self.log(f'val_{coef_name}', outputs[coef_name],sync_dist=True)
        
        self.log(f'val_loss', loss_val,sync_dist=True)
        
        pred = outputs['pred']
        errors = get_errors( pred, outputs['texts'])
        
        for error_name, error in errors.items():
            self.log(f'val_{error_name}', error,sync_dist=True)
        return loss_val
    
    def test_step(self, batch, batch_idx):
        outputs = self.net(**batch,do_pred=True)
        pred = outputs['pred']
        errors = get_errors( pred, outputs['texts'], average=False)
        return errors
    
    def test_epoch_end(self,outputs):
        agg_outputs = {'cer':[],'wer':[]}
        for output in outputs:
            agg_outputs['cer']+=output['cer']
            agg_outputs['wer']+=output['wer']
        self.test_results= agg_outputs
        return agg_outputs
    
                      
    def configure_optimizers(self):
        
        opt_fun = torch.optim.Adam
        opt = opt_fun(self.net.parameters(),lr=self.lr,eps=1e-4)
        
        if self.use_cosine_lr:
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,eta_min=self.lr*.1,T_max=self.T_max)
            return [opt], [{"scheduler": sch, "interval": "step"}]
        else:
            return [opt], []