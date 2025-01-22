from src.ljspeech import SpeechDataModule
from src.vocoder_trainer import VocoderTrainer
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
import hydra
from hydra.utils import get_original_cwd
from pathlib import Path
import torch

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

@hydra.main(config_path='configs', config_name='lj_hificar_dur')
def main(cfg):
    
    # adjusting relative path to absolute path
    cfg['data']['wav_dir'] = fix_path(cfg['data']['wav_dir'])
    cfg['data']['unit_dir'] = fix_path(cfg['data']['unit_dir'])
    cfg['data']['split_manifests']['train'] = fix_path(cfg['data']['split_manifests']['train'])
    cfg['data']['split_manifests']['dev'] = fix_path(cfg['data']['split_manifests']['dev'])
    cfg['data']['split_manifests']['test'] = fix_path(cfg['data']['split_manifests']['test'])
    cfg['generator_params']['emb_p'] = fix_path(cfg['generator_params']['emb_p'])
    
    # datamodule
    datamodule = SpeechDataModule(**cfg['data'])
    
    # model
    model = VocoderTrainer(cfg)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    if 'experiment_name' in cfg.keys() and cfg['experiment_name'] is not None:
        save_dir = cfg['experiment_name']        
    else:
        save_dir = None

    # checkpoint best
    checkpoint_callback_topk = ModelCheckpoint(
        monitor="mel_loss",
        save_top_k=1,
        mode="min",
        filename='best-{epoch}-{val_loss:.2f}'
    )
    
    # checkpoint every N epochs
    checkpoint_callback_by_epoch = ModelCheckpoint(
        every_n_epochs=cfg['checkpoint_epoch'],
    )
    callbacks  = [checkpoint_callback_topk, 
                  LearningRateMonitor(logging_interval='step'), 
                  checkpoint_callback_by_epoch]
    
    
    # Trainer
    if cfg['gpus'] is not None:
        gpus = [int(x) for x in cfg['gpus'].split(',')]
    else:
        gpus= None
    
    
    trainer = pl.Trainer(devices=gpus,
                         accelerator="gpu",
                         #strategy="ddp",
                         strategy='ddp_find_unused_parameters_true',
                         max_steps = cfg['max_steps'],
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=cfg['check_val_every_n_epoch'],
                         limit_val_batches=cfg['limit_val_batches'],
                         callbacks=callbacks,
                         accumulate_grad_batches=cfg['accumulate_grad_batches'],
                         default_root_dir=save_dir,
                        )

    # fit model
    trainer.fit(model, datamodule, ckpt_path=cfg['resume_ckpt'])

if __name__ =='__main__':
    main()
