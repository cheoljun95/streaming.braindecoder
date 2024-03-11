# check source code https://pytorch.org/audio/main/_modules/torchaudio/models/rnnt.html#RNNT
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchaudio.models.rnnt import _Transcriber
from typing import List, Optional, Tuple
from model import tokenizer
from pathlib import Path


class GaussianSmoother(nn.Module):
    
    def __init__(self,window_size, smoothing_sigma, input_dim, stride,padding,**kwargs):
        super().__init__()
        support = torch.arange(-(window_size-1)//2, (window_size-1)//2 + 1, dtype=torch.float)
        kernel = Normal(loc=0, scale=smoothing_sigma).log_prob(support).exp_()
        kernel_1d = kernel.mul_(1 / kernel.sum())
        kernel_1d = kernel_1d.repeat(input_dim).reshape(input_dim,-1)
        self.conv = nn.Conv1d(input_dim, input_dim, window_size, stride=stride, padding=padding, groups=input_dim)
        self.conv.weight.data = kernel_1d.unsqueeze(1)
        self.conv.requires_grad_(False)
        self.left_pad = window_size//2-padding//2
        
    def forward(self, x,input_len):
        return self.conv(x.permute(0,2,1)).permute(0,2,1),input_len-self.left_pad
    
    
class CNNLSTMTranscriber(nn.Module, _Transcriber):
    def __init__(self,input_dim, output_dim, conv_channel_dims=[512,512], conv_kernel_sizes=[3,3], conv_strides=[2,2],
                lstm_num_layers=3, lstm_hidden_dim=512, dropout=0.2, bidirectional=True, use_res=False,use_res_proto=False, use_layernorm=False,
                smth_configs=None, input_spatial_norm=False,use_gru=False, random_context_clip_num=0,random_context_clip_prob=0.0,
                ablate_regions=[],ablate_mask_path=None):
        super().__init__()
        
        
        
        self.input_spatial_norm=input_spatial_norm
        if smth_configs is not None:
            self.smooth = GaussianSmoother(**smth_configs)
        else:
            self.smooth = None
        cnn_block = ResBlock if use_res else Block
        if use_res_proto:
            cnn_block = ResBlockProto
        self.conv = [cnn_block(input_dim, conv_channel_dims[0], conv_kernel_sizes[0], conv_strides[0])]
        for i in range(1,len(conv_channel_dims)):
            self.conv.append(cnn_block(conv_channel_dims[i-1],conv_channel_dims[i], conv_kernel_sizes[i], conv_strides[i]))
        
        self.conv = nn.Sequential(*self.conv)
        self.fc = nn.Sequential(nn.Linear(conv_channel_dims[-1], lstm_hidden_dim),
                                nn.Dropout(dropout))
        if use_layernorm:
            self.lstm = LayerNormLSTM(lstm_hidden_dim, lstm_hidden_dim, lstm_num_layers,dropout=dropout)
        elif use_gru:
            self.lstm = nn.GRU(lstm_hidden_dim, lstm_hidden_dim, lstm_num_layers, batch_first=True, dropout=dropout,
                               bidirectional=bidirectional,)
        else:
            self.lstm = nn.LSTM(lstm_hidden_dim, lstm_hidden_dim, lstm_num_layers, batch_first=True, dropout=dropout,
                               bidirectional=bidirectional,)
        self.ds_factor = np.prod(conv_strides)
        
        lstm_output_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim
        self.logit = nn.Sequential(nn.Linear(lstm_output_dim,lstm_output_dim),
                                   nn.Dropout(dropout),
                                   nn.Linear(lstm_output_dim,output_dim))
        self.output_dim = output_dim
        self.random_context_clip_prob = random_context_clip_prob
        self.random_context_clip_num = random_context_clip_num
        if len(ablate_regions)>0:
            self.ablate = AblateChannels(ablate_regions,mask_path = ablate_mask_path)
        else:
            self.ablate = None
        
    def extract(self, input, lengths): 
        
        if self.ablate is not None:
            input = self.ablate(input)
        lengths = lengths//self.ds_factor
        #input = input[:,:int((input.shape[1]//self.ds_factor)*self.ds_factor)]
        if self.input_spatial_norm:
            std = input[:,:,:253].std(-1)[:,:,None]
            std[std==0] = 1
            input[:,:,:253] = (input[:,:,:253]-input[:,:,:253].mean(-1)[:,:,None])/std
            std = input[:,:,253:].std(-1)[:,:,None]
            std[std==0] = 1
            input[:,:,253:] = (input[:,:,253:]-input[:,:,253:].mean(-1)[:,:,None])/std
            
        features = input.permute(0, 2, 1)
        if self.smooth is not None:
            features = self.smooth(features)
        features = self.conv(features).permute(0, 2, 1)
        features = self.fc(features) 
        
        return features, lengths
    
    def forward(self, input, lengths, output_features=False): 
        
        if self.ablate is not None:
            input = self.ablate(input)
        lengths = lengths//self.ds_factor
        #input = input[:,:int((input.shape[1]//self.ds_factor)*self.ds_factor)]
        if self.input_spatial_norm:
            std = input[:,:,:253].std(-1)[:,:,None]
            std[std==0] = 1
            input[:,:,:253] = (input[:,:,:253]-input[:,:,:253].mean(-1)[:,:,None])/std
            std = input[:,:,253:].std(-1)[:,:,None]
            std[std==0] = 1
            input[:,:,253:] = (input[:,:,253:]-input[:,:,253:].mean(-1)[:,:,None])/std
            
        features = input.permute(0, 2, 1)
        if self.smooth is not None:
            features = self.smooth(features)
        features = self.conv(features).permute(0, 2, 1)
        features = self.fc(features) 
        if self.random_context_clip_num>0 and np.random.uniform()< self.random_context_clip_prob:
            clip_num = np.random.choice(self.random_context_clip_num,1)[0]+1
            clip_points = np.random.choice(lengths.min().cpu().item()-clip_num, clip_num)
            clip_points.sort()
            clip_points = clip_points + np.arange(clip_num) +1
            outputs_ = [self.lstm(features[:,:clip_points[0]])[0]]
            for p in range(1, len(clip_points)):
                outputs_.append(self.lstm(features[:,clip_points[p-1]:clip_points[p]])[0])
                
            outputs_.append(self.lstm(features[:,clip_points[-1]:])[0])
            outputs = torch.cat(outputs_,1)
        else:
            outputs, _ = self.lstm(features)
        logits = self.logit(outputs)
        
        if output_features:
            return logits, lengths,features
        else:
            return logits, lengths
    
    def infer(self, input, lengths, states,):
        if self.ablate is not None:
            input = self.ablate(input)
        lengths = lengths//self.ds_factor
        #input = input[:,:int((input.shape[1]//self.ds_factor)*self.ds_factor)]
        features = input.permute(0, 2, 1)
        if self.smooth is not None:
            features = self.smooth(features)
        features = self.conv(features).permute(0, 2, 1)
        features = self.fc(features)
        outputs, states = self.lstm(features,states)
        logits = self.logit(outputs)
        
        return logits, lengths, states

BaseTranscriber = CNNLSTMTranscriber

class DummyTranscriber(nn.Module, _Transcriber):
    def __init__(self,input_dim):
        super().__init__()
        self.ds_factor = 1
        self.output_dim =input_dim
        
    def forward(self, input, lengths, **kwargs): 
        return input, lengths
    
    def infer(self, input, lengths, states,):
        return input, lengths, None
    
class AblateChannels(nn.Module):
    def __init__(self, ablate_regions, mask_path):
        super().__init__()
        if mask_path is None:
            mask_path = Path(__file__).parent.parent/'misc'/'ablation_maps.npy'
        ablation_maps = np.load(mask_path, allow_pickle=True)[()]
        mask = np.ones(506)>0
        for region in ablate_regions:
            mask = mask & ablation_maps[region]
        
        self.mask = mask
    def forward(self, x):
        mask = torch.zeros_like(x)
        mask[...,self.mask] = 1
        x = x*mask
        return x
    
class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, kernel_size=3,stride=1, dilation=1):
        super().__init__()
        
        if isinstance(kernel_size, int):
            k1 = kernel_size
            k2 = kernel_size
        else:
            k1 = kernel_size[0]
            k2 = kernel_size[1]
        
        p1 = k1 //2
        p2 = (k2 //2)*dilation
        self.conv1 = nn.Conv1d(num_ins, num_outs, k1, padding=p1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs) 
        self.conv2 = nn.Conv1d(num_outs, num_outs, k2, padding=p2, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(num_outs) 
        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs) 
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)
    
class Block(nn.Module):
    def __init__(self, num_ins, num_outs, kernel_size=3,stride=1, dilation=1):
        super().__init__()
        
        
        self.conv = nn.Conv1d(num_ins, num_outs, kernel_size, padding=0, stride=stride,dilation=dilation)
        self.bn = nn.BatchNorm1d(num_outs) 

    def forward(self, x):
        input_value = x
        x = F.relu(self.bn(self.conv(x)))
        return x