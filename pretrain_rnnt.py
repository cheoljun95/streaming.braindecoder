from dataloader.datamodule import SpeechTextDataModule
from model.unit_rnnt import SpeechModel
import lightning as pl
from lightning.callbacks import LearningRateMonitor,ModelCheckpoint,EarlyStopping, Callback
import hydra
from pathlib import Path
import torch
from utils.utils import load_model

torch.set_float32_matmul_precision('medium')

@hydra.main(config_path='configs')
def main(cfg):
    
    #torch.manual_seed(int(cfg['seed']))
    
    # datamodule
    datamodule = SpeechTextDataModule(**cfg['data'])

    # model
    if 'resume_ckpt_path' in cfg.keys() and cfg['resume_ckpt_path'] is not None:
        model, _ = load_model(version_dir=cfg['resume_ckpt_path'],MODEL=SpeechModel,
                              mode='latest', verbose=True)
    else:
        model = SpeechModel(**cfg['model'])
                       
    # Callbacks
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
    # Trainer
    if cfg['gpus'] is not None:
        gpus = [int(x) for x in cfg['gpus'].split(',')]
    else:
        gpus= None
    
    callbacks  = [checkpoint_callback_topk, checkpoint_callback_by_epoch,
                  LearningRateMonitor(logging_interval='step')]
    
    if 'earlystop_metric' in cfg.keys() and cfg['earlystop_metric'] is not None:
        early_stop_callback = EarlyStopping(monitor=cfg['earlystop_metric'], min_delta=0.00, patience=30, verbose=False, mode="min")
        callbacks.append(early_stop_callback)
    scaler = torch.cuda.amp.GradScaler()
    
    trainer = pl.Trainer(devices=gpus,
                         accelerator="gpu",
                         strategy="ddp",
                         max_steps = cfg['max_steps'],
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=cfg['check_val_every_n_epoch'],
                         limit_val_batches=cfg['limit_val_batches'],
                         callbacks=callbacks,
                         accumulate_grad_batches=cfg['accumulate_grad_batches'],
                         gradient_clip_val=0.5,
                         default_root_dir=save_dir
                        )

    # fit model
    trainer.fit(model,datamodule)

if __name__ =='__main__':
    main()
