import torch
from collections import OrderedDict
import numpy as np
import argparse
from pathlib import Path
from utils.utils import load_cfg_and_ckpt_path

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--mode", type=str, default='best')
parser.add_argument("--term", type=str, default='mean_uer')

if __name__ == '__main__':
    args = parser.parse_args()
    version_dir = args.ckpt_path
    cfg, ckpt_path = load_cfg_and_ckpt_path(version_dir=version_dir, mode=args.mode, term=args.term)
    state_dict= torch.load(ckpt_path,map_location='cpu')['state_dict']
    exp_name = cfg['experiment_name']

    module_state_dict = OrderedDict()
    for module_name, state in state_dict.items():
        if f'net.' in module_name:
            new_name = module_name.split(f'net.')[-1]
            module_state_dict[new_name] = state

    module_ckpt = {'config': cfg['model'],
                   'state_dict': module_state_dict}
    torch.save(module_ckpt, args.output_path)