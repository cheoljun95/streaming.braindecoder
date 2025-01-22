import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from pathlib import Path
import random
import torch.nn as nn
import soundfile as sf
import librosa


PAD_ID = 100 # hard-coded as using 100 KMean clusters

def deduplicate(units):
    units = np.concatenate([units, np.array([PAD_ID+1])])
    changes = (units[1:] != units[:-1])*1.0
    changes = np.nonzero(changes)[0]
    unique_units = units[changes]
    unitlens = np.concatenate([changes[:1]+1, np.diff(changes)])
    return unique_units, unitlens

class SpeechDataset(Dataset):
    
    def __init__(self, wav_dir, unit_dir, tags, sample_len, ar_len):
        super().__init__()
        self.wav_dir = wav_dir
        self.unit_dir = unit_dir
        self.tags = tags
        self.sample_len = sample_len
        self.ar_len = ar_len
        assert self.sample_len % 320 == 0
        
    def __len__(self):
        return len(self.tags)
    
    def __getitem__(self,i):
        tag = self.tags[i]
        wav_file = self.wav_dir/f"{tag}.wav"
        unit_file = self.unit_dir/f"{tag}.npy"
        
        wav, orig_sr = sf.read(wav_file)
        
        if orig_sr != 16000:
            wav = librosa.resample(wav, orig_sr=orig_sr,
                                   target_sr=16000)
        sr = 16000
        units = np.load(unit_file)
        wav = wav[:len(units)*320]
        
        unit_sample_len = self.sample_len // 320
        if len(wav) < self.sample_len:
            wav = np.concatenate([wav, np.zeros(self.sample_len-len(wav))])
            units = np.concatenate([units, np.array([PAD_ID] * (unit_sample_len-len(units)))])
            ar = np.zeros(self.ar_len)
        else:
            p = np.random.randint(0, len(units)-unit_sample_len)
            units = units[p:p+unit_sample_len]
            wav = wav[p*320:p*320+self.sample_len]
            ar = wav[max(0,p*320-self.ar_len):p*320]
            if len(ar) < self.ar_len:
                ar = np.concatenate([np.zeros(self.ar_len-len(ar)), ar])
        
        wav = torch.from_numpy(wav).float()
        
        units, dur = deduplicate(units)
        units = torch.from_numpy(units).long()
        dur = torch.from_numpy(dur).long()
        ar = torch.from_numpy(ar).float()
        
        return {'wav':wav, 'units':units, 'ar':ar, 'dur':dur}
    
    @staticmethod
    def collate(batch):
        data = {}
        data['wav'] = nn.utils.rnn.pad_sequence([d['wav'] for d in batch], batch_first=True, padding_value=0.0)
        data['ar'] = nn.utils.rnn.pad_sequence([d['ar'] for d in batch], batch_first=True, padding_value=0.0)
        data['units'] = nn.utils.rnn.pad_sequence([d['units'] for d in batch], batch_first=True, padding_value=PAD_ID)
        data['dur'] = nn.utils.rnn.pad_sequence([d['dur'] for d in batch], batch_first=True, padding_value=0)
        
        return data
    

class SpeechDataModule(LightningDataModule):
    def __init__(self,
                 wav_dir,
                 unit_dir,
                 split_manifests,
                 sample_len,
                 ar_len,
                 batch_size,
                 num_workers=4,
                 drop_last=True,
                 pin_memory=True,
                 ):
        super().__init__()
        
        self.wav_dir = Path(wav_dir)
        self.unit_dir = Path(unit_dir)
        self.split_manifests = split_manifests
        self.batch_size=batch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_batch_size = batch_size 
        self.sample_len = sample_len 
        self.ar_len = ar_len
        
    def _load_tags(self, split):
        with open(self.split_manifests[split], "r") as f:
            tags = [Path(l.rstrip()).stem for l in f.readlines()]
            tags = [tag for tag in tags if (self.wav_dir/f"{tag}.wav").exists() and (self.unit_dir/f"{tag}.npy").exists()]
        return tags
    
    def train_dataloader(self,):
        dataset = SpeechDataset(self.wav_dir, self.unit_dir, self._load_tags("train"),
                                self.sample_len, self.ar_len)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size ,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=SpeechDataset.collate
        )
        return loader
    
    def val_dataloader(self):
        dataset = SpeechDataset(self.wav_dir, self.unit_dir, self._load_tags("dev"),
                                self.sample_len, self.ar_len)
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=SpeechDataset.collate
        )
        return loader
    
    def test_dataloader(self):
        dataset = SpeechDataset(self.wav_dir, self.unit_dir, self._load_tags("test"),
                                self.sample_len, self.ar_len)
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=SpeechDataset.collate
        )
        return loader
    