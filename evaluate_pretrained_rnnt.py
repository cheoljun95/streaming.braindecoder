from data.datamodule import SpeechTextDataModule
from model.unit_rnnt import SpeechModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint,EarlyStopping, Callback
import hydra
from pathlib import Path
import torch
import argparse
from utils.utils import load_cfg_and_ckpt_path
import json
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    output_path = args.output_path
    cfg, ckpt_path = load_cfg_and_ckpt_path(output_path, mode='latest',verbose=False)
    cfg['model']['step_max_tokens']=100 
    cfg['model']['beam_width'] = 10
    model = SpeechModel(**cfg['model'])
    cfg['data']['batch_size'] = 20
    datamodule = SpeechTextDataModule(**cfg['data'])
    dataloader = datamodule.test_dataloader()
    dataloader.dataset.data=dataloader.dataset.data[:100]
    trainer = pl.Trainer(accelerator='gpu', devices=[0])
    trainer.test(model, dataloader,ckpt_path=ckpt_path)
    results = model.test_results
    errors = {metric:np.mean(result) for metric,result in results.items()}
         
    with open(output_path+"/test_scores.json", "w") as outfile:
        json.dump(errors, outfile)

    print(errors)
'''
from data.datamodule import SpeechTextDataModule
from model.unit_rnnt import SpeechModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint,EarlyStopping, Callback
import hydra
from pathlib import Path
import torch
import argparse
from utils.utils import load_model
import json
from utils.evaluate import get_errors


parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str)

if __name__ == '__main__':
    device='cuda'
    args = parser.parse_args()
    output_path = args.output_path
    model, cfg = load_model(output_path, mode='latest',verbose=False)
    model = model.net.eval().to(device)
    
    datamodule = SpeechTextDataModule(**cfg['data'])
    dataloader = datamodule.test_dataloader()
    
    for batch in dataloader:
        batch = {_:d.to(device) for _,d in batch.items()}
        outputs = model(**batch)
        pred = outputs['pred']
        batch_errors = get_errors( pred, batch['texts'])
        
    errors = {'cer':0,
             'wer':0}
    
    for n, result in enumerate(results):
        for error_key, error in errors.items():
            errors[error_key] = (errors[error_key]*n+error)/(n+1)
         
    with open(output_path+"/test_scores.json", "w") as outfile:
        json.dump(errors, outfile)

    
    

'''
