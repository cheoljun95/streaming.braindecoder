import torch
from pathlib import Path
from model.ecog2speech_rnnt import SpeechModel
import hydra
import tqdm
import numpy as np
from hydra.utils import get_original_cwd
import re, string
regex = re.compile('[%s]' % re.escape(string.punctuation))

def remove_dropout(cfg, name):
    if isinstance(cfg, dict):
        for subname, subdict in cfg.items():
            cfg[subname] = remove_dropout(subdict, subname)
    elif 'drop' in name:
        return 0.0
    else:
        return cfg
    return cfg


def parse_data(files, data_dir):
    data = []
    with open(files, "r") as f:
        for line in f.readlines():
            filename, text, hb_unit = line.split("|")
            text = regex.sub('', text.lower())
            text = ' '.join([t for t in text.split(' ') if t != ''])
            data.append({"ecog": data_dir/filename,
                         "phoneme": text,
                         "unit": hb_unit})
    return data
    
def fix_path(path):
    if str(path)[0] !='/':
        return Path(get_original_cwd())/path
    else:
        return path
    
@hydra.main(config_path='eval_configs')
def main(cfg):
    
    device = cfg['device']
    
    # set save path
    save_path = fix_path(cfg['save_path'])
    save_path.mkdir(exist_ok=True)
    save_path = save_path/'salience'
    save_path.mkdir(exist_ok=True)
    
    # load model
    # Dropout is disabled
    rnnt_ckpt_path = fix_path(cfg['rnnt_ckpt_path'])
    ckpt = torch.load(rnnt_ckpt_path, map_location='cpu')
    model_config = ckpt['config']
    model_config = remove_dropout(model_config, '')
    model = SpeechModel(**model_config).net
    model.load_state_dict(ckpt['state_dict'])
    model = model.train().to(device)
    
    # load dataset
    test_files = fix_path(cfg['test_files'])
    data_dir = fix_path(cfg['data_dir'])
    test_data = parse_data(test_files, data_dir)
    

    for di, b in tqdm.tqdm(enumerate(test_data)):
        x = test_data[di]
        original_ecog = torch.from_numpy(np.load(x['ecog'])).float().to(device).unsqueeze(0)
        inputs = {'ecogs':original_ecog}
        inputs['texts'] = {"unit": [x['unit']], "phoneme":[x['phoneme']]}
        inputs['ecog_lens'] = torch.tensor([inputs['ecogs'].shape[1]]).to(device)
        
        original_losses = {}
        total_loss = 0
        for target_name in model.names:
            outputs = model(**inputs)
            total_loss+=outputs[f'{target_name}_rnnt_loss']
        total_loss.backward()
        
        original_rnnt_loss = total_loss.detach().cpu().item()
        
        salience = np.zeros(253)
        for ei in range(253):
            perturb_mask = torch.ones_like(inputs['ecogs'][:,:,:])
            perturb_mask[:,:,ei] = 0
            perturb_mask[:,:,ei+253] = 0
            perturbed_ecogs = inputs['ecogs']*perturb_mask
            outputs = model(ecogs=perturbed_ecogs, ecog_lens = inputs['ecog_lens'],texts= inputs['texts'])
            perturbed_total_loss = 0
            for target_name in model.names:
                perturbed_loss =  outputs[f'{target_name}_rnnt_loss'].detach().cpu().item()
                perturbed_total_loss+=perturbed_loss
            
            salience[ei] = perturbed_total_loss-original_rnnt_loss
        
        np.save(save_path/f'{di}.npy', salience)
    
if __name__ =='__main__':
    main()