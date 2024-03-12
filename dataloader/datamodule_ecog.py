import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from pathlib import Path
import random, string, re
from torchvision import transforms
regex = re.compile('[%s]' % re.escape(string.punctuation))
    

class ECoGDataset(Dataset):
    
    def __init__(self, data, transform):
        super().__init__()
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,i):
        data = self.data[i]
        output = {}
        output["ecog"] = torch.from_numpy(np.load(data["ecog"])).float()
        if self.transform is not None:
            output["ecog"] = self.transform(output["ecog"])
                    
        output['ecog_len'] = len(output["ecog"])
        output["text"] = {"unit":data["unit"],
                          "phoneme":data["phoneme"]}
        return output
    
    @staticmethod
    def collate(batch):
        data = {}
        data['ecogs'] = nn.utils.rnn.pad_sequence([d['ecog'] for d in batch], batch_first=True, padding_value=0.0)
        data['ecog_lens'] = torch.tensor([d['ecog_len'] for d in batch])
        data['texts']={entry: [d['text'][entry] for d in batch] for entry in batch[0]['text'].keys()}
        return data
    
class ECoGDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 train_files,
                 val_files = None,
                 trainval_ratio=[0.95,0.05],
                 shuffle_trainval_split=False, 
                 batch_size=64,
                 val_batch_size=None,
                 transform_config={},
                 num_workers=4,
                 drop_last=True,
                 pin_memory=True,
                 no_transform=False,
                 no_val_transform=False,
                 **kwargs,
                 ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.train_files = self._parse_files(train_files)
        self.val_files = val_files
        if self.val_files is None:
            if shuffle_trainval_split:
                random.shuffle(self.train_files)
            self.train_files, self.val_files = (self.train_files[:int(len(self.train_files)*trainval_ratio[0])],
                            self.train_files[-int(len(self.train_files)*trainval_ratio[1]):])
        else:
            self.val_files = self._parse_files(val_files)
        self.batch_size=batch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.transform_config = transform_config
        self.no_transform = no_transform
        self.no_val_transform=no_val_transform
        
    def _parse_files(self, files):
        data = []
        with open(files, "r") as f:
            for line in f.readlines():
                filename, text, hb_unit = line.split("|")
                text = regex.sub('', text.lower())
                text = ' '.join([t for t in text.split(' ') if t != ''])
                data.append({"ecog": self.data_dir/filename,
                             "phoneme": text,
                             "unit": hb_unit})
        return data
                             
    def train_dataloader(self):
        dataset = ECoGDataset(self.train_files, transform=self.get_transform('train',**self.transform_config))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=ECoGDataset.collate
        )
        return loader
    
    def val_dataloader(self):
        transform = None if self.no_val_transform else self.get_transform('train',**self.transform_config)
        dataset = ECoGDataset(self.val_files,transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=ECoGDataset.collate
        )
        return loader
    
    def get_transform(self,mode='train',
                     jitter_range=[0.8, 1.0],
                      jitter_max_start=200,
                     channeldropout_prob=0.5,
                     channeldropout_rate=0.2,
                     scaleaugmnet_range = [0.95,1.05],
                      sample_window=32,
                     transform_list=['jitter', 'channeldrop','scale'],
                      no_jitter=False,
                      cutdim=253,
                      **kwargs
                     ):
        
        if self.no_transform:
            return None
        
        if mode =='train':
            transforms_=[]
            if 'cutdim' in transform_list:
                transforms_.append(CutDim(cutdim))
            if 'jitter' in transform_list and not no_jitter:
                transforms_.append(Jitter(jitter_range,jitter_max_start))
            if 'channeldrop' in transform_list:
                transforms_.append(ChannelDrop(channeldropout_prob, channeldropout_rate))
            if 'scale' in transform_list:
                transforms_.append(ScaleAugment(scaleaugmnet_range))
            if 'sample_window' in transform_list:
                transforms_.append(SampleWindow(sample_window))
                
            return transforms.Compose(transforms_)
        else:
            return None
      
    
class CutDim(object):
    
    def __init__(self, cutdim=253):
        self.cutdim = cutdim
        
    def __call__(self, x):
        return x[:,:self.cutdim]
    
class ChannelDrop(object):
    
    def __init__(self, apply_prob=0.5, dropout=0.2):
        self.apply_prob = apply_prob
        self.dropout = dropout 
        try:
            self.is_range = len(dropout)>1
        except:
            self.is_range = False
        
    def __call__(self, x):
        if random.uniform(0, 1) < self.apply_prob:
            if self.is_range:
                dropout = np.random.uniform(self.dropout[0],self.dropout[1])
            else:
                dropout = self.dropout
            drop_mask = np.random.uniform(size=x.shape[1])<dropout
            x[:, drop_mask] = 0
        return x

class Jitter(object):
    def __init__(self, fraction_range=[0.8,1.0],max_start=100):
        self.max_start = 400
        self.fraction_pool = np.linspace(fraction_range[0],fraction_range[1],5)
    def __call__(self, x):
        
        fraction = np.random.choice(self.fraction_pool, 1)[0]
        start_f = np.random.uniform()*(1-fraction)
        end_f = start_f+fraction
        si,ei = int(len(x)*start_f),max(len(x),len(x)*end_f)
        si = min(self.max_start,si)
        x=x[si:ei]
        return x
    
    
class ScaleAugment(object):
    def __init__(self, range_):
        self.range = range_
        
    def __call__(self, x):
        scale = np.random.uniform(self.range[0], self.range[1],size=x.shape[1])
        x= x*scale[None,:]
        return x
    
    
class SampleWindow(object):
    def __init__(self, window_size):
        self.window_size = window_size
        
    def __call__(self, x):
        onset = np.random.uniform(0,len(x)-self.window_size)
        onset = min(len(x)-self.window_size, int(onset))
        
        x= x[onset:onset+self.window_size]
        return x

    
