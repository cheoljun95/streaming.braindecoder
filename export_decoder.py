import torch
from collections import OrderedDict
import numpy as np
import argparse
from pathlib import Path
from utils.utils import load_cfg_and_ckpt_path
import datetime

def get_timestamp(path):
    time = path.parent.stem
    date = path.parent.parent.stem
    timestamp = datetime.datetime(*[int(s) for s in date.split('-')],
                                  *[int(s) for s in time.split('-')])
    return timestamp


def check_output(path,min_epoch=1):
    ckpts = [ckpt for ckpt in path.glob("**/*.ckpt") if 'best' not in ckpt.stem]
    valid = len(ckpts)>0
    if valid:
        epoch = int(ckpts[0].stem.split('epoch=')[-1].split('-')[0])
        valid = valid & (epoch>=min_epoch)
        
    return valid
    

def get_latest_valid_dir(exp_name="joint_hbbpe_tm1k", output_dir="outputs/"):
    output_dir = Path(output_dir)
    min_epoch = 4
    
    dirs = [d for d in output_dir.glob(f"**/{exp_name}") if check_output(d)]
    
    dirs.sort(key= lambda v: get_timestamp(v))
    
    latest_valid_dir = dirs[-1]
    
    return str(latest_valid_dir)


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default="model_ckpts/tm1k.ckpt")
parser.add_argument("--mode", type=str, default='best')
parser.add_argument("--term", type=str, default='mean_uer')

if __name__ == '__main__':
    args = parser.parse_args()
    version_dir = args.ckpt_path
    if version_dir is None:
        version_dir = get_latest_valid_dir(exp_name="joint_hbbpe_tm1k", output_dir="outputs/")
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