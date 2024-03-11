# check source code https://pytorch.org/audio/main/_modules/torchaudio/models/rnnt.html#RNNT
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchaudio.models.rnnt import _Predictor

BasePredictor = _Predictor

'''
class CustomPredictor(nn.Module, _Predictor):
    pass 
'''

class EmbPredictor(nn.Module):
    def __init__(self, num_symbols, output_dim, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(num_symbols, output_dim)
    
    def forward(self, input, lengths,state=None):
        
        return self.emb(input),lengths, []
    

    
class CausalCNNPredictor(nn.Module):
    def __init__(self, num_symbols, output_dim,kernels,cnn_channel, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(num_symbols, output_dim)
        cnn = [CausalConv1d(output_dim,cnn_channel,kernels[0],padding=kernels[0]//2)]
        for i in range(1,len(kernels)):
            cnn+=[CausalConv1d(cnn_channel,cnn_channel,kernels[i],padding=kernels[i]//2)]
        self.conv = nn.Sequential(*cnn)
        self.logit = nn.Sequential(nn.Linear(cnn_channel, output_dim),
                                   nn.GELU(),
                                   nn.Linear(output_dim, output_dim),
                                   nn.LayerNorm(output_dim))
                  
    
    def forward(self, input, lengths,state=None):
        
        x = self.emb(input) # B, L, d
        x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        x = self.logit(x)
        
        return x,lengths, []
    
    
class CausalConv1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        
    def forward(self, x):
        self.conv.weight.data[:,:,self.conv.weight.data.shape[-1]//2+1:] =0.0
        return self.conv(x)
        
    