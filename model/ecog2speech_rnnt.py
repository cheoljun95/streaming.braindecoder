import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from lightning import LightningModule
except:
    LightningModule = object
import math
import random
from torchaudio.models.rnnt import emformer_rnnt_model, RNNT
from torchaudio.transforms import RNNTLoss
from torchaudio.models.rnnt_decoder import RNNTBeamSearch
from .modules import build_predictor, build_joiner, build_transcriber, build_feature_extractor
from .tokenizer import BPETokenizer, HuBERTTokenizer, PhonemeTokenizer, GraphemeTokenizer
from utils.utils import anneal_coef
from utils.evaluate import get_unit_errors
import numpy as np

class MunitRNNT(nn.Module):
    """
    """
    def __init__(self, feature_extractor_configs, bundle_configs,tokenizer_configs=None, beam_width=10,
                 step_max_tokens=100, **kwargs):
        
        super().__init__()
        
        self.feature_extractor = build_transcriber(**feature_extractor_configs) 
        self.rnnts =[]
        self.tokenizers = []
        self.token_decoders = []
        self.remove_tokens_list = []
        self.search_list = []
        self.targets = []
        self.blank_ids = []
        self.losses = []
        self.names = []
        for module_name, config_bundle in bundle_configs.items():
            if 'target' in config_bundle.keys():
                target = config_bundle['target']
            else:
                target = module_name
            transcriber = build_transcriber(**config_bundle['transcriber_configs'])
            predictor = build_predictor(**config_bundle['predictor_configs'])
            joiner = build_joiner(**config_bundle['joiner_configs'])
            rnnt = RNNT(transcriber, 
                         predictor, 
                         joiner)
            
            self.rnnts.append(rnnt)
        
            blank_id=config_bundle['predictor_configs']['num_symbols']-1
            if target == 'unit':
                if 'tokenizer_type' in config_bundle.keys():
                    if tokenizer_type == 'HB':
                        tokenizer = HuBERTTokenizer(**config_bundle['tokenizer_configs'])
                    else:
                        raise NotImplementedError
                else:
                    tokenizer = HuBERTTokenizer(**config_bundle['tokenizer_configs'])
            else:
                if 'tokenizer_type' in config_bundle.keys():
                    tokenizer_type = config_bundle['tokenizer_type']
                    if tokenizer_type == 'BPE':
                        tokenizer = BPETokenizer(blank_id)
                    elif tokenizer_type == 'PH':
                        tokenizer = PhonemeTokenizer(**config_bundle['tokenizer_configs'])
                    elif tokenizer_type == 'GR':
                        tokenizer = GraphemeTokenizer(**config_bundle['tokenizer_configs'])
                    else:
                        raise NotImplementedError
                else:
                    tokenizer = PhonemeTokenizer(**config_bundle['tokenizer_configs'])
            self.tokenizers.append(tokenizer)
            
            self.token_decoders.append(tokenizer.get_decoder())
            self.remove_tokens_list.append(tokenizer.get_remove_tokens())
            self.search_list.append(RNNTBeamSearch(rnnt, blank_id,step_max_tokens=step_max_tokens))
            self.targets.append(target)
            self.names.append(module_name)
            self.blank_ids.append(blank_id)
            loss=RNNTLoss(reduction='sum',blank=blank_id)
            self.losses.append(loss)
        
        self.rnnts = nn.ModuleList(self.rnnts)
        self.beam_width = beam_width
    
    
    def forward(self,ecogs, ecog_lens, texts, do_pred=False, **kwargs):
        """
        """
        
        inputs = ecogs
        input_lens = ecog_lens
        sources, source_lengths = self.feature_extractor(inputs, input_lens)
        
        outputs = {}
        source_lengths=source_lengths.clip(0,sources.shape[1])
        input_source_lengths = source_lengths.to(sources.device)
        
        for module_name, target, rnnt, tokenizer, remove_tokens, search, token_decoder, loss in zip(self.names, self.targets,self.rnnts,
                                                                                               self.tokenizers,self.remove_tokens_list,
                                                                                              self.search_list,self.token_decoders, self.losses):
            
            with torch.no_grad():    
                targets, target_lengths, gt_texts = tokenizer(texts=texts[target], wavs=None)
            
            source_encodings, orig_source_lengths = rnnt.transcriber(
                input=sources,
                lengths=input_source_lengths,
            )
            orig_source_lengths = orig_source_lengths.clip(0,source_encodings.shape[1])
                
            targets = targets.to(sources.device).to(torch.int32)
            target_legnths = target_lengths.to(sources.device)
            target_encodings, target_lengths, predictor_state = rnnt.predictor(
                input=targets,
                lengths=target_lengths,
            )
            output, source_lengths, target_lengths = rnnt.joiner(
                source_encodings=source_encodings,
                source_lengths=orig_source_lengths,
                target_encodings=target_encodings,
                target_lengths=target_lengths,
            )
            target_lengths = target_lengths.to(sources.device).to(torch.int32)
            source_lengths = source_lengths.to(torch.int32).clip(0,output.shape[1])
            loss =loss(output,targets[:,1:].contiguous(),source_lengths,target_lengths-1).mean() 
            if do_pred:
                transcribed_outputs=source_encodings
                with torch.no_grad():
                    decoded=[search._search(trans[None,:l], None, beam_width=self.beam_width)[0][0] for trans,l in zip(transcribed_outputs, source_lengths)]
                pred = [token_decoder([t for t in dec if t not in remove_tokens+[tokenizer.blank]])
                             for dec in decoded]
            else:
                pred = []
            
            outputs[f'{module_name}_rnnt_loss'] = loss
            outputs[f'{module_name}_logits'] = output
            outputs[f'{module_name}_logit_lengths']=source_lengths
            outputs[f'{module_name}_targets'] = targets
            outputs[f'{module_name}_target_lengths']= target_lengths
            outputs[f'{module_name}_pred'] = pred
            outputs[f'{module_name}_texts'] = gt_texts
            
        return outputs
    
    
class SpeechModel(LightningModule):

    def __init__(self, loss_coef_instructions, non_val_loss=[], lr=0.001,use_cosine_lr=True, T_max=500000, skip_pred=False, **model_configs):
        super().__init__()

        self.loss_coef_dict = loss_coef_instructions
        self.lr = lr
        self.net = MunitRNNT(**model_configs)
        self.non_val_loss = non_val_loss
        self.T_max = T_max
        self.use_cosine_lr = use_cosine_lr
        self.test_results = None
        self.skip_pred = skip_pred
            
    def forward(self, **kwargs):
        return self.net(**kwargs)

    def _update_coef(self):
        for coef_name, coef_inst in self.loss_coef_dict.items():
            if 'anneal_step_range' in coef_inst.keys() and coef_inst['anneal_step_range'] is not None:
                coef_inst['value'] = anneal_coef(self.global_step, coef_inst['anneal_step_range'],
                                                 coef_inst['anneal_value_range'])
                self.log(f'{coef_name}_coef', coef_inst['value'])
    
        
        
    def training_step(self, batch, **kwargs):
        self._update_coef()
        outputs = self.net(**batch)
        loss_val = 0
        for coef_name, coef_inst in self.loss_coef_dict.items():
            if coef_name in outputs.keys() and coef_inst['value']>0:
                loss_val += coef_inst['value'] * outputs[coef_name]
                self.log(f'train_{coef_name}', outputs[coef_name])
        self.log(f'train_loss', loss_val)
        return loss_val

            
            
    def validation_step(self, batch, **kwargs):
        outputs = self.net(**batch, do_pred=True and not self.skip_pred)
        loss_val = 0
        
        for coef_name, coef_inst in self.loss_coef_dict.items():
            if coef_name in outputs.keys():
                if coef_name not in self.non_val_loss:
                    loss_val += coef_inst['value'] * outputs[coef_name]
                self.log(f'val_{coef_name}', outputs[coef_name],sync_dist=True)
        
        self.log(f'val_loss', loss_val,sync_dist=True)
        if not self.skip_pred:
            uers=[]
            for target in self.net.names:
                pred = outputs[f'{target}_pred']
                errors = get_unit_errors( pred, outputs[f'{target}_texts'])

                for error_name, error in errors.items():
                    self.log(f'val_{target}_{error_name}', error,sync_dist=True)
                    uers.append(error)
            mean_uer = np.mean(uers)
            self.log(f'val_mean_uer', mean_uer,sync_dist=True)
        return loss_val
                      
    def configure_optimizers(self):
        
        opt_fun = torch.optim.Adam
        opt = opt_fun(self.net.parameters(),lr=self.lr,eps=1e-4)
        
        if self.use_cosine_lr:
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,eta_min=self.lr*.1,T_max=self.T_max)
            return [opt], [{"scheduler": sch, "interval": "step"}]
        else:
            return [opt], []