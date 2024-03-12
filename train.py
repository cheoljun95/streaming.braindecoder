from dataloader.datamodule_ecog import ECoGDataModule
from model.ecog2speech_rnnt import SpeechModel
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, Callback
import hydra
from hydra.utils import get_original_cwd
from pathlib import Path
import torch
from utils.utils import load_cfg_and_ckpt_path

torch.set_float32_matmul_precision('medium')

def config_get(cfg, entry, default):
    if entry not in cfg.keys():
        return default
    else:
        return cfg[entry]

def fix_path(path):
    if str(path)[0] !='/':
        return Path(get_original_cwd())/path
    else:
        return path

@hydra.main(config_path='configs', config_name='tm1k')
def main(cfg):
    
    # datamodule
    cfg['data']['data_dir'] = fix_path(cfg['data']['data_dir'])
    cfg['data']['train_files'] = fix_path(cfg['data']['train_files'])
    if cfg['data']['val_files'] is not None:
        cfg['data']['val_files'] = fix_path(cfg['data']['val_files'])
    datamodule = ECoGDataModule(**cfg['data'])

    # model
    model = SpeechModel(**cfg['model'])
    
    predictor_ckpts = config_get(cfg, 'predictor_ckpt', None)
    if predictor_ckpts is not None:
        for mi, predictor_ckpt in enumerate(predictor_ckpts):
            if predictor_ckpt is None:
                continue
            print(f"Loading predictor from {str(fix_path(predictor_ckpt))}")
            predictor_ckpt = torch.load(fix_path(predictor_ckpt))
            model.net.rnnts[mi].predictor.load_state_dict(predictor_ckpt['state_dict'])
            if config_get(cfg['model'], 'freeze_predictor', False):
                model.net.rnnts[mi].predictor.requires_grad_(False)
    
    joiner_ckpts = config_get(cfg, 'joiner_ckpt', None)
    if joiner_ckpts is not None:
        for mi, joiner_ckpt in enumerate(joiner_ckpts):
            if joiner_ckpt is None:
                continue
            print(f"Loading joiner from {str(fix_path(joiner_ckpt))}")
            joiner_ckpt = torch.load(fix_path(joiner_ckpt))
            model.net.rnnts[mi].joiner.load_state_dict(joiner_ckpt['state_dict'],strict=False)
            if config_get(cfg['model'], 'freeze_joiner', False):
                model.net.rnnts[mi].joiner.requires_grad_(False)
                
    resume_ckpt = config_get(cfg, 'resume_ckpt', None)
    if resume_ckpt is not None:
        mode =  config_get(cfg, 'resume_mode', 'latest')
        _, resume_ckpt = load_cfg_and_ckpt_path(version_dir=cfg['resume_ckpt'], mode='latest')
        
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    if 'experiment_name' in cfg.keys() and cfg['experiment_name'] is not None:
        save_dir = cfg['experiment_name']        
    else:
        save_dir = None

    # checkpoint best
    checkpoint_callback_topk = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename='best-{epoch}-{val_loss:.2f}'
    )
    
    # checkpoint every N epochs
    checkpoint_callback_by_epoch = ModelCheckpoint(
        every_n_epochs=cfg['checkpoint_epoch'],
    )
    callbacks  = [checkpoint_callback_topk, LearningRateMonitor(logging_interval='step'), checkpoint_callback_by_epoch] #, early_stop_callback]
    
    
    if 'log_metrics' in cfg.keys():
        for metric in cfg['log_metrics']:
            callbacks.append(ModelCheckpoint(
                monitor=f"val_{metric}",
                save_top_k=1,
                mode="min",
                filename='bestmeanuer-{epoch}-{val_'+metric+':.2f}'
            ))
    else:
        for metric in ['bpe_uer','mean_uer']:
            callbacks.append(ModelCheckpoint(
                monitor=f"val_{metric}",
                save_top_k=1,
                mode="min",
                filename='bestmeanuer-{epoch}-{val_'+metric+':.2f}'
            ))
        
    if 'earlystop_metric' in cfg.keys() and cfg['earlystop_metric'] is not None:
        patience = 100
        if 'patience' in cfg.keys():
            patience=cfg['patience'] 
        early_stop_callback = EarlyStopping(monitor=cfg['earlystop_metric'], min_delta=config_get(cfg,"min_delta",0.01),
                                            patience=patience, verbose=False, mode="min")
        callbacks.append(early_stop_callback)

    # Trainer
    if cfg['gpus'] is not None:
        gpus = [int(x) for x in cfg['gpus'].split(',')]
    else:
        gpus= None
    
    
    trainer = pl.Trainer(devices=gpus,
                         accelerator="gpu",
                         #strategy="ddp",
                         strategy='ddp_find_unused_parameters_true',
                         max_epochs = cfg['max_epochs'],
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=cfg['check_val_every_n_epoch'],
                         limit_val_batches=cfg['limit_val_batches'],
                         callbacks=callbacks,
                         accumulate_grad_batches=cfg['accumulate_grad_batches'],
                         gradient_clip_val=cfg['gradient_clip_val'],
                         default_root_dir=save_dir
                        )

    # fit model
    trainer.fit(model, datamodule, ckpt_path=resume_ckpt)

if __name__ =='__main__':
    main()
