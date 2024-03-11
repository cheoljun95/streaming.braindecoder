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
    
    def __init__(self, data, text_names, **kwargs):
        super().__init__()
        self.data = data
        self.text_names = text_names
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,i):
        data_path, texts = self.data[i][0], self.data[i][1:]
        x = torch.from_numpy(np.load(data_path)).float()
        output = {'input':x, 'input_len':len(x)}
        for text, text_name in zip(texts, self.text_names):
            output[text_name] = text
        
        return output
    
    @staticmethod
    def collate(batch):
        data = {}
        text_names = [key for key in batch[0].keys() if 'wav' not in key]
        data['texts'] = {text_name: [d[text_name] for d in batch] for text_name in text_names}
        data['inputs'] = nn.utils.rnn.pad_sequence([d['input'] for d in batch], batch_first=True, padding_value=0.0)
        data['input_lens'] = torch.tensor([d['input_len'] for d in batch])
        return data
    
                  
        

class SpeechTextDataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir,
                 transcriptions=['transcription','hubert-l6_km100_transcription'],
                 ft_name='features/psuedo-ema_wavlm_large-l9',
                 text_names=['text','unit'],
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
        self.ft_name = ft_name
        self.transcriptions = transcriptions
        self.batch_size=batch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.do_regularize = [trans=='transcription' for trans in self.transcriptions]
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
        self.text_names = text_names
        
    def _load_data(self, split):
        split_names={'train':  ['train-clean-100', 'train-clean-360', 'train-other-500'],
                    'valid':['dev-clean'],
                    'test':['test-clean','test-other']}[split]
        
        data = []
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        for split_name in split_names:
            tag2texts = []
            valid_tags = None
            wav_lens = None
            for trans, do_regularize in zip(self.transcriptions,self.do_regularize):
                texts=[]
                with open(str(self.root_dir/trans/f'{split_name}.transcription.txt'), 'r') as f:
                    texts = f.readlines()
                with open(str(self.root_dir/trans/f'{split_name}.tag.txt'), 'r') as f:
                    tags = f.readlines()
                
                texts = [text.rstrip() for text in texts]
                if do_regularize:
                    texts = [regex.sub('', text.lower()) for text in texts]
                    
                tags=[]
                with open(str(self.root_dir/trans/f'{split_name}.tag.txt'), 'r') as f:
                    tags = f.readlines()
                    
                if trans=='transcription':
                    wav_lens = [int(tag.rstrip().split(' ')[0]) for tag in tags]
                tags = [tag.rstrip().split(' ')[-1] for tag in tags]
                
                if trans=='transcription':
                    wav_lens = {tag:len_ for tag,len_ in zip(tags, wav_lens)}
                #tag2text = {tag:text for tag,text in zip(tags, texts)}
                tag2texts.append({tag:text for tag,text in zip(tags, texts)})
                if valid_tags is None:
                    valid_tags = set(tags)
                else:
                    valid_tags = valid_tags & set(tags)
            
            valid_tags = list(valid_tags)
                
            for tag in valid_tags:
                if wav_lens is not None and wav_lens[tag]>16000*10:
                    continue
                data_path = self.root_dir/self.ft_name/f'{tag}.npy'
                if data_path.exists():
                    data.append([str(data_path)]+[tag2text[tag] for tag2text in tag2texts])
                else:
                    print(f"{str(data_path)} doesn't exist")
        print(f'@@@@@@@@ Total {len(data)} data points for {split} @@@@@@@@@')
        assert len(data)>0
        return data
    
    
    
    def train_dataloader(self) -> DataLoader:
        
        data = self._load_data('train')
        dataset = SpeechTextDataset(data,text_names=self.text_names)
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
        dataset = SpeechTextDataset(data,text_names=self.text_names)
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
    
