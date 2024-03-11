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
import pandas as pd
from torchvision import transforms
from scipy.signal import butter, lfilter, filtfilt, decimate

def normalize(x, axis=-1, order=2):
    """Normalizes a Numpy array.
    Args:
        x: Numpy array to normalize.
        axis: axis along which to normalize.
        
        order: Normalization order (e.g. `order=2` for L2 norm).
    Returns:
        A normalized copy of the array.
    """
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)
    
    
        
def butter_bandpass(cut, fs, order=5):
    if isinstance(cut,list) and len(cut) == 2:
        return butter(order, cut, fs=fs, btype='bandpass')
    else:
        return butter(order, cut, fs=fs, btype='low')

def butter_bandpass_filter(data, cut, fs, order=5):
    b, a = butter_bandpass(cut, fs, order=order)
    y = filtfilt(b, a, data,axis=0)
    return y


class ECoGPhonemeDataset(Dataset):
    
    def __init__(self, data, transform=None, do_preprocess=True):
        super().__init__()
        self.data = data
        phoneme_file_path = Path(__file__).parent.parent/'misc'/'tokens_phrases_1k.txt'
        with open(phoneme_file_path, 'r') as f:
            self.phonemes = [ph.rstrip() for ph in f.readlines()]
        lexicon_file_path = Path(__file__).parent.parent/'misc'/'lexicon_phrases_1k.txt'   
        
        self.word2phs={}
        with open(lexicon_file_path, 'r') as f:
            for lexicon in f.readlines():
                lexicon = lexicon.rstrip()
                lexicon_parsed = lexicon.split(' ') #last is '|'
                self.word2phs[lexicon_parsed[0]]=lexicon_parsed[1:]
        self.ph2idx = {ph:i for i, ph in enumerate(self.phonemes)}
        self.transform = transform
        self.do_preprocess = do_preprocess
        
        
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def _preprocess(hga):
        # hga: (T, d=506)
        # low-pass filter and down-sample
        hga_lp = butter_bandpass_filter(hga,16.67,200)
        hga_lpds = decimate(hga_lp, 6, axis=0)
        # L2 unit normalization
        '''
        hga_norm = np.linalg.norm(hga_lpds[:,203:],axis=1,keepdims=True)
        hga_norm[hga_norm==0]=1
        lf_norm = np.linalg.norm(hga_lpds[:,203:],axis=1,keepdims=True)
        lf_norm[lf_norm==0]=1
        hga_lpds[:,:203] = hga_lpds[:,:203]/hga_norm
        hga_lpds[:,203:] = hga_lpds[:,203:]/lf_norm
        '''
        hga_lpds = normalize(hga_lpds,0)
        return hga_lpds
    
    def __getitem__(self,i):
        ecog_path, text = self.data[i]
        ecog = np.load(ecog_path)
        if self.do_preprocess:
            ecog = self._preprocess(ecog)
        if self.transform is not None:
            ecog = self.transform(ecog)
        ecog = torch.from_numpy(ecog.copy()).float()
        phonemes = ['|']
        for lexicon in text.split(' '):
            phonemes += self.word2phs[lexicon]
        phonemes.append('|')
        ph_idxs = [self.ph2idx[ph] for ph in phonemes]
        ph_idxs = torch.tensor(ph_idxs).long()
        output = {'ecog': ecog,
                  'ecog_len': len(ecog),
                  'phoneme': ph_idxs,
                  'phoneme_len': len(ph_idxs),
                  'text':text}
        
        return output
    
    @staticmethod
    def collate(batch):
        data = {}
        data['ecogs'] = nn.utils.rnn.pad_sequence([d['ecog'] for d in batch], batch_first=True, padding_value=0.0)
        data['ecog_lens'] = torch.tensor([d['ecog_len'] for d in batch])
        data['phonemes'] = nn.utils.rnn.pad_sequence([d['phoneme'] for d in batch], batch_first=True, padding_value=-1)
        data['phoneme_lens'] = torch.tensor([d['phoneme_len'] for d in batch])
        data['texts']=[d['text'] for d in batch]
        return data
    
class ECoGPhonemeDataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir,
                 metainfo_dir=None,
                 paradigm='tm1k',
                 batch_size=64,
                 trainval_ratio=[0.9,0.1],
                 ecog_fileprefix='hgr_',
                 do_preprocess=True,
                 transform_config={},
                 val_batch_size=None,
                 num_workers=4,
                 drop_last=True,
                 pin_memory=True,
                 no_transform=False,
                 relabeled=False,
                 ):
        super().__init__()
        
        
        self.root_dir = Path(root_dir)
        self.paradigm = paradigm
        if metainfo_dir is not None:
            metainfo_dir = Path(metainfo_dir)
        else:
            metainfo_dir = self.root_dir/paradigm
            
        df = pd.read_csv(str(metainfo_dir/f'{paradigm}_dataframe.csv'))
        
        train_fn, test_fn = df['train_filename'][0], df['test_filename'][0]
        with open(metainfo_dir/train_fn,'r') as f:
            self.train_files = [fn.rstrip() for fn in f.readlines()]
        with open(metainfo_dir/test_fn,'r') as f:
            self.test_files = [fn.rstrip() for fn in f.readlines()]
        
        
        
        self.train_files, self.val_files = (self.train_files[:int(len(self.train_files)*trainval_ratio[0])],
                            self.train_files[-int(len(self.train_files)*trainval_ratio[1]):])
        
        self.idx2text={}
        self.idx2fileidx={}
        for fileidx, (block_idx, trial_idx, text) in enumerate(zip(df['block_num'].values,
                                                  df['trial_num'].values,
                                                  df['gt_text'].values)):
            idx = f'B{block_idx}_{trial_idx:05d}'
            if idx in self.idx2text.keys():
                assert self.idx2text[idx]==text
            self.idx2text[idx]=text
            self.idx2fileidx[idx]=fileidx
        
        self.ecog_fileprefix = ecog_fileprefix
        self.do_preprocess = do_preprocess
        self.batch_size=batch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.transform_config = transform_config
        self.no_transform = no_transform
        
    def _load_data(self, split):
        
        data_files = {'train':self.train_files,
                      'val':self.val_files,
                      'test':self.test_files}[split]
        data = []
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        for data_file in data_files:
            idx = '_'.join(data_file.split('_')[1:3])
            text= self.idx2text[idx]
            text = regex.sub('', text.lower())
            text = ' '.join([t for t in text.split(' ') if t != ''])
            data.append([self.root_dir/self.paradigm/f'{self.ecog_fileprefix}{self.idx2fileidx[idx]}.npy', text])
        return data
    
    
    
    def train_dataloader(self) -> DataLoader:
        
        data = self._load_data('train')
        dataset = ECoGPhonemeDataset(data, transform=self.get_transform('train',**self.transform_config),
                                    do_preprocess=self.do_preprocess)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=ECoGPhonemeDataset.collate
        )
        return loader
    
    def val_dataloader(self) -> DataLoader:
        
        data = self._load_data('val')
        dataset = ECoGPhonemeDataset(data, transform=self.get_transform('test',**self.transform_config),
                                    do_preprocess=self.do_preprocess)
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=ECoGPhonemeDataset.collate
        )
        return loader
    
    def test_dataloader(self) -> DataLoader:
        
        data = self._load_data('test')
        dataset = ECoGPhonemeDataset(data, transform=self.get_transform('test',**self.transform_config),
                                    do_preprocess=self.do_preprocess)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory,
            collate_fn=ECoGPhonemeDataset.collate
        )
        return loader
    
    def get_transform(self,mode='train',
                     jitter_range=[0.8, 1.0],
                     channeldropout_prob=0.5,
                     channeldropout_rate=0.2,
                     scaleaugmnet_range = [0.95,1.05],
                     transform_list=['jitter', 'channeldrop'],
                      no_jitter=False,
                      **kwargs
                     ):
        
        if self.no_transform:
            return None
        
        if mode =='train':
            transforms_=[]
            if 'jitter' in transform_list and not no_jitter:
                transforms_.append(Jitter(jitter_range))
            if 'channeldrop' in transform_list:
                transforms_.append(ChannelDrop(channeldropout_prob, channeldropout_rate))
            if 'scale' in transform_list:
                transforms_.append(ScaleAugment(scaleaugmnet_range))
            return transforms.Compose(transforms_)
        else:
            jitter = Jitter(jitter_range)
            if no_jitter:
                return None
            else:
                return transforms.Compose([jitter])
      
    

    
class ChannelDrop(object):
    
    def __init__(self, apply_prob=0.5, dropout=0.2):
        self.apply_prob = apply_prob
        self.dropout = dropout 
        
    def __call__(self, x):
        if random.uniform(0, 1) < self.apply_prob:
            drop_mask = np.random.uniform(size=x.shape[1])<self.dropout
            x[:, drop_mask] = 0
        return x

class Jitter(object):
    def __init__(self, fraction_range=[0.8,1.0]):
        self.fraction_pool = np.linspace(fraction_range[0],fraction_range[1],5)
    def __call__(self, x):
        
        fraction = np.random.choice(self.fraction_pool, 1)[0]
        start_f = np.random.uniform()*(1-fraction)
        end_f = start_f+fraction
        si,ei = int(len(x)*start_f),max(len(x),len(x)*end_f)
        x=x[si:ei]
        return x
    
class ScaleAugment(object):
    def __init__(self, range_):
        self.range = range_
        
    def __call__(self, x):
        scale = np.random.uniform(self.range[0], self.range[1],size=x.shape[1])
        x= x*scale[None,:]
        return x
