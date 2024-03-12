# check source code https://pytorch.org/audio/main/_modules/torchaudio/models/rnnt.html#RNNT
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchaudio.models.rnnt import _Joiner

BaseJoiner = _Joiner


class PremapJoiner(torch.nn.Module):

    def __init__(self, input_dim, output_dim, activation= "tanh", hidden_dim=None):
        super().__init__()
        
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation {activation}")
            
        hidden_dim = input_dim if hidden_dim is None else hidden_dim                       
        
        self.f_target = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.GELU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim))
        self.linear = torch.nn.Linear(hidden_dim, output_dim, bias=True)    

    def forward(
        self,
        source_encodings,
        source_lengths,
        target_encodings,
        target_lengths,
    ):
        
        #source_encodings = self.f_source(source_encodings)
        target_encodings = self.f_target(target_encodings)
        joint_encodings = source_encodings.unsqueeze(2).contiguous() + target_encodings.unsqueeze(1).contiguous()
        activation_out = self.activation(joint_encodings)
        output = self.linear(activation_out)
        return output, source_lengths, target_lengths
    
    
class PremapJoinerv2(torch.nn.Module):

    def __init__(self,transcriber_dim, input_dim, output_dim, activation= "tanh", hidden_dim=None,
                fix_linear=False):
        super().__init__()
        
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation {activation}")
            
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        '''
        self.f_source = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.GELU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim))
        '''                            
        
        self.premap = torch.nn.Linear(transcriber_dim, input_dim, bias=True)    
        self.linear = torch.nn.Linear(hidden_dim, output_dim, bias=True)    
        self.fix_linear = fix_linear

    def forward(
        self,
        source_encodings,
        source_lengths,
        target_encodings,
        target_lengths,
    ):
        if self.fix_linear:
            self.linear.requires_grad_(False)
        source_encodings = self.premap(source_encodings)
        joint_encodings = source_encodings.unsqueeze(2).contiguous() + target_encodings.unsqueeze(1).contiguous()
        activation_out = self.activation(joint_encodings)
        output = self.linear(activation_out)
        return output, source_lengths, target_lengths
    
    
    