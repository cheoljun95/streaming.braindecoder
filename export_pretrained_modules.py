import torch
from collections import OrderedDict
from model.modules import build_predictor, build_joiner, build_transcriber, build_feature_extractor
from utils.utils import load_cfg_and_ckpt_path
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--output_path", type=str, default='ckpts')

if __name__ == '__main__':
    args = parser.parse_args()
    version_dir = args.ckpt_path
    output_dir = Path(args.output_path)
    output_dir.mkdir(exist_ok=True)
    cfg, ckpt_path = load_cfg_and_ckpt_path(version_dir=version_dir, mode='latest')
    state_dict= torch.load(ckpt_path,map_location='cpu')['state_dict']
    exp_name = cfg['experiment_name']
    for module_key in ['predictor', 'joiner', 'transcriber']:

        module_state_dict = OrderedDict()
        for module_name, state in state_dict.items():
            if f'net.{module_key}' in module_name:
                new_name = module_name.split(f'net.{module_key}.')[-1]
                module_state_dict[new_name] = state

        cfg['model'][f'{module_key}_configs']

        module_ckpt = {'config': cfg['model'][f'{module_key}_configs'],
                       'state_dict': module_state_dict}
        torch.save(module_ckpt, output_dir/f'{exp_name}_{module_key}.ckpt')