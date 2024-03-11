import numpy as np
import torch
import yaml
from pathlib import Path


def anneal_coef(step, anneal_step_range, anneal_value_range):

    step = min(max(anneal_step_range[0], step), anneal_step_range[1])
    ratio = (step - anneal_step_range[0]) / (anneal_step_range[1] - anneal_step_range[0])
    annealed = anneal_value_range[0] - (anneal_value_range[0] - anneal_value_range[1]) * ratio

    return annealed

def load_model(version_dir, MODEL, mode='best',verbose=False,term='best'):
    version_dir = Path(version_dir)
    # get configuration information
    try:
        cfg = yaml.load(open(version_dir / '.hydra' / 'config.yaml'), Loader=yaml.FullLoader)
    except:
        cfg = yaml.load(open(version_dir.parent / '.hydra' / 'config.yaml'), Loader=yaml.FullLoader)
    version_name = [f for f in (version_dir / 'lightning_logs').glob('version_*')][0].stem
    if mode == 'best':
        checkpoint_path = [f for f in Path(version_dir / 'lightning_logs' / version_name / 'checkpoints').glob('*.ckpt')
                           if 'best' in f.name and term in f.name]
        checkpoint_path = checkpoint_path[-1]
    else:
        checkpoint_path = [f for f in
                           Path(version_dir / 'lightning_logs' / version_name / 'checkpoints').glob('*.ckpt')]

        def get_epoch(fileName):
            epoch = [n for n in fileName.split('-') if 'epoch' in n][0]
            return int(epoch.split('=')[-1])

        checkpoint_path.sort(key=lambda f: get_epoch(f.name))
        checkpoint_path = checkpoint_path[-1]
        
    if verbose:
        print(f'Model is loaded from {str(checkpoint_path)}')
    model = MODEL(**cfg['model'])
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=False)
    print(str(checkpoint_path))
    model = model.eval()

    return model, cfg

def load_cfg_and_ckpt_path(version_dir, mode='best',verbose=False,term='best'):
    version_dir = Path(version_dir)
    # get configuration information
    try:
        cfg = yaml.load(open(version_dir / '.hydra' / 'config.yaml'), Loader=yaml.FullLoader)
    except:
        cfg = yaml.load(open(version_dir.parent / '.hydra' / 'config.yaml'), Loader=yaml.FullLoader)
    version_name = [f for f in (version_dir / 'lightning_logs').glob('version_*')][0].stem
    if mode == 'best':
        checkpoint_path = [f for f in Path(version_dir / 'lightning_logs' / version_name / 'checkpoints').glob('*.ckpt')
                           if 'best' in f.name and term in f.name]
        checkpoint_path = checkpoint_path[-1]
    else:
        checkpoint_path = [f for f in
                           Path(version_dir / 'lightning_logs' / version_name / 'checkpoints').glob('*.ckpt')]

        def get_epoch(fileName):
            epoch = [n for n in fileName.split('-') if 'epoch' in n][0]
            return int(epoch.split('=')[-1])

        checkpoint_path.sort(key=lambda f: get_epoch(f.name))
        checkpoint_path = checkpoint_path[-1]
        
    print(str(checkpoint_path))
    return cfg, str(checkpoint_path)