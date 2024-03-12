import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import tqdm
import random
import csv
import soundfile as sf
import string
import re

from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

class SpeechTextDataset(Dataset):
    
    def __init__(self, data, skip_text=False, max_len=None, label_sr=50):
        super().__init__()
        self.data = data
        self.skip_text = skip_text
        self.max_len = max_len
        self.label_sr = label_sr
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,i):
        flac_path, text = self.data[i]
        wav,sr = sf.read(flac_path)
        if self.max_len is not None:
            wav = wav[:int(self.max_len*16000)]
            text = ' '.join(text.split(' ')[:int(self.max_len*self.label_sr)])
        assert sr ==16000
        output = {'wav':wav,
                  'text': text}
        
        return output
    
    @staticmethod
    def collate(batch):
        data = {}
        data['texts'] = [d['text'] for d in batch]
        
        wav_input = processor([d['wav'] for d in batch],
                                   sampling_rate=16000, return_tensors="pt",
                                   padding=True)
        
        data['wavs'] = wav_input.input_values.detach()
        data['wav_lens'] = np.array([len(d['wav']) for d in batch])
        return data
    
                  
        

class SpeechTextDataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir,
                 transcription='transctiption',
                 labellen_file=None,
                 labellen_thr=None,
                 max_len=None,
                 label_sr=None,
                 batch_size=64,
                 val_batch_size=None,
                 num_workers=4,
                 drop_last=True,
                 pin_memory=True,
                 
                 ):
        super().__init__()
        
        
        self.root_dir = Path(root_dir)
        self.transcription = transcription
        self.batch_size=batch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.do_regularize = self.transcription=='transcription'
        if labellen_file is not None:
            with open(labellen_file, 'r') as f:
                len_tags =f.readlines()
            self.tag2labellen = {}
            for len_tag in len_tags:
                len_,tag = len_tag.split(' ')
                tag = tag.rstrip()
                self.tag2labellen[tag]=int(len_)
        else:
            self.tag2labellen = None
        self.labellen_thr= labellen_thr
        self.max_len = None
        self.label_sr = None
        
    def _load_data(self, split):
        split_names={'train':  ['train-clean-100', 'train-clean-360', 'train-other-500'],
                    'valid':['dev-clean'],
                    'test':['test-clean','test-other']}[split]
        
        data = []
        for split_name in split_names:
            texts=[]
            with open(str(self.root_dir/self.transcription/f'{split_name}.transcription.txt'), 'r') as f:
                texts = f.readlines()
            texts = [text.rstrip() for text in texts]

            tags=[]
            with open(str(self.root_dir/self.transcription/f'{split_name}.tag.txt'), 'r') as f:
                tags = f.readlines()
            tags = [tag.rstrip() for tag in tags]
            regex = re.compile('[%s]' % re.escape(string.punctuation))
            for tag,text in zip(tags, texts):
                len_, tag=tag.split(' ')
                if self.labellen_thr is not None:
                    if self.tag2labellen[tag]> self.labellen_thr:
                        continue
                if int(len_)>200000:
                    continue
                if self.do_regularize:
                    text = regex.sub('', text.lower())
                data.append([str(self.root_dir/tag)+'.flac', text])
        return data
    
    
    
    def train_dataloader(self) -> DataLoader:
        
        data = self._load_data('train')
        dataset = SpeechTextDataset(data,max_len=self.max_len,label_sr=self.label_sr)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=SpeechTextDataset.collate
        )
        return loader
    
    def val_dataloader(self) -> DataLoader:
        
        data = self._load_data('valid')
        dataset = SpeechTextDataset(data,max_len=self.max_len,label_sr=self.label_sr)
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=SpeechTextDataset.collate
        )
        return loader
    
    def test_dataloader(self) -> DataLoader:
        
        data = self._load_data('test')
        dataset = SpeechTextDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory,
            collate_fn=SpeechTextDataset.collate
        )
        return loader
    
