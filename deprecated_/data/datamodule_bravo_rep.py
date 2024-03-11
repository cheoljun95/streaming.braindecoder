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
import librosa
import string
import re
import pandas as pd
from torchvision import transforms
from scipy.signal import butter, lfilter, filtfilt, decimate
from transformers import Wav2Vec2Processor

#processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
regex = re.compile('[%s]' % re.escape(string.punctuation))
        
def butter_bandpass(cut, fs, order=5):
    if isinstance(cut,list) and len(cut) == 2:
        return butter(order, cut, fs=fs, btype='bandpass')
    else:
        return butter(order, cut, fs=fs, btype='low')

def butter_bandpass_filter(data, cut, fs, order=5):
    b, a = butter_bandpass(cut, fs, order=order)
    y = filtfilt(b, a, data,axis=0)
    return y


class ECoGDataset(Dataset):
    
    def __init__(self, data,transform=None, only_hga=False, **kwargs):
        super().__init__()
        self.data = data
        self.labels = list(data.keys())
        self.transform = transform
        self.only_hga =only_hga
        
    def __len__(self):
        return 5000 #len(self.data)
    
    
    def __getitem__(self,i):
        ecog1, ecog2 = random.sample(self.data[self.labels[i%len(self.data.keys())]],2)
        
        output={}
        for ei,ecog in zip(['ecog1','ecog2'],[ecog1,ecog2]):
            ecog = np.load(ecog)
            if self.transform is not None:
                assert ecog is not None
                ecog = self.transform(ecog)
            ecog = torch.from_numpy(ecog).float()
            if self.only_hga:
                ecog = ecog[:,:253]
            output[f'{ei}'] = ecog
            output[f'{ei}_len'] = len(ecog)
                
        return output
    
    @staticmethod
    def collate(batch):
        data = {}
        data['ecogs'] = nn.utils.rnn.pad_sequence([d['ecog1'] for d in batch], batch_first=True, padding_value=0.0)
        data['ecog_lens'] = torch.tensor([d['ecog1_len'] for d in batch])
        data['ecogs2'] = nn.utils.rnn.pad_sequence([d['ecog2'] for d in batch], batch_first=True, padding_value=0.0)
        data['ecog_lens2'] = torch.tensor([d['ecog2_len'] for d in batch])  
        return data
    
class ECoGDataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir,
                 hb_dir,
                 ft_dir=None,
                 metainfo_dir=None,
                 audio_dir='/data/cheoljun/b3_audio_scale-2/',
                 metainfo_path=None,
                 textidx_file=None,
                 paradigm='tm1k',
                 batch_size=64,
                 trainval_ratio=[0.95,0.05],
                 ecog_fileprefix='hgr_',
                 transform_config={},
                 decimate=1,
                 val_batch_size=None,
                 num_workers=4,
                 drop_last=True,
                 pin_memory=True,
                 include_audio=False,
                 include_ecog=False,
                 include_unit=False,
                 include_phoneme=False,
                 include_feature=False,
                 include_block=False,
                 shuffle_trainval_split=False,
                 train_files = None,
                 test_files = None,
                 val_files = None,
                 load_list=[],
                 text_list=[],
                 no_transform=False,
                 use_test_for_valid=False,
                 block_idxs=None,
                 only_hga=False,
                 relabeled = False,
                 bn2date= '/data/cheoljun/b3_misc/bn2date.npy',
                 convert_bn2date=False,
                 **kwargs,
                 ):
        super().__init__()
        
        self.audio_dir = Path(audio_dir)/paradigm
        if textidx_file is None:
            textidx_file = self.audio_dir/'file2textindex.npy'
        
        self.ft_dir = Path(ft_dir)/paradigm.split('_recent')[0] if ft_dir is not None else None
        self.hb_dir = Path(hb_dir)/paradigm.split('_recent')[0] 
        self.fileidx2textidx = np.load(textidx_file, allow_pickle=True)[()]
        self.root_dir = Path(root_dir)
        self.paradigm = paradigm
        self.convert_bn2date =convert_bn2date
        if metainfo_dir is not None:
            metainfo_dir = Path(metainfo_dir)
        else:
            metainfo_dir = self.root_dir/paradigm
        
        if metainfo_path is None:
            
            df = pd.read_csv(str(metainfo_dir/f'{paradigm}_dataframe.csv'))
        else:
            df = pd.read_csv(metainfo_path)
        if train_files is None:
            train_files = metainfo_dir/df['train_filename'][0]
        
        if test_files is None:
            test_files = metainfo_dir/df['test_filename'][0]
        #self.train_files.sort()
        #train_fn, test_fn = df['train_filename'][0], df['test_filename'][0]
        with open(train_files,'r') as f:
            self.train_files = [fn.rstrip() for fn in f.readlines()]
        with open(test_files,'r') as f:
            self.test_files = [fn.rstrip() for fn in f.readlines()]
        
        self.train_files.sort()
        
        if val_files is None:
            if not use_test_for_valid:
                if shuffle_trainval_split:
                    assert False
                    random.shuffle(self.train_files)

                self.train_files, self.val_files = (self.train_files[:int(len(self.train_files)*trainval_ratio[0])],
                                self.train_files[-int(len(self.train_files)*trainval_ratio[1]):])
            else:
                self.val_files = self.test_files
        else:
            print('@@@@@@@@ loading validation files @@@@@@@@@@')
            with open(val_files,'r') as f:
                self.val_files = [fn.rstrip() for fn in f.readlines()]
                
        self.bn2date = np.load(bn2date, allow_pickle=True)[()] if self.convert_bn2date else None
        
        train_blocks = [int(f.split('_')[1][1:]) for f in self.train_files]
        val_blocks = [int(f.split('_')[1][1:]) for f in self.val_files]
        if self.convert_bn2date:
            train_blocks = [self.bn2date[bn] for bn in train_blocks]
            val_blocks = [self.bn2date[bn] for bn in val_blocks]
        
        print(f'@@@@@@@@ {len(set(train_blocks)&set(val_blocks))} blocks are overlapping in train and val! @@@@@@@')
        self.idx2text={}
        self.idx2fileidx={}
        self.idx2filename={}
        for fileidx, (block_idx, trial_idx, text, label) in enumerate(zip(df['block_num'].values,
                                                  df['trial_num'].values,
                                                  df['gt_text'].values,
                                                                         df['gt_idx'])):
            idx = f'B{block_idx}_{trial_idx:05d}'
            if idx in self.idx2text.keys():
                assert self.idx2text[idx]==text
            self.idx2text[idx]=text
            self.idx2fileidx[idx]=fileidx
            self.idx2filename[fileidx] = f'{ecog_fileprefix}{block_idx}_{trial_idx}_{label}'
        
        self.ecog_fileprefix = ecog_fileprefix
        self.batch_size=batch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.transform_config = transform_config
        self.decimate = decimate
        #self.transform_config['decimation']=decimate
        
        self.include_audio = include_audio
        self.include_ecog = include_ecog
        self.include_unit = include_unit
        self.include_phoneme = include_phoneme
        self.include_feature = include_feature
        self.include_block = include_block
        self.load_list = load_list
        self.text_list = text_list
        self.no_transform = no_transform
        if include_block:
            if block_idxs is None:
                blocks = np.unique(df['block_num'].values).astype(int)
                if self.convert_bn2date:
                    blocks = np.unique([self.bn2date[bn] for bn in blocks])
                blocks.sort()
                print(f'@@@@@@@@ Total {len(blocks)} blocks are found @@@@@@@@@')
                self.block_idxs = {block:bi for bi,block in enumerate(blocks)}
            else:
                self.block_idxs = np.load(block_idxs, allow_pickle=True)[()]
        self.only_hga = only_hga
        self.relabeled=relabeled
        
    def _load_data(self, split):
        
        data_files = {'train':self.train_files,
                      'val':self.val_files,
                      'test':self.test_files}[split]
        data = {}
        for data_file in data_files:
            idx = '_'.join(data_file.split('_')[1:3])
            if self.relabeled:
                ecog_path = self.root_dir/self.paradigm/f'{self.idx2filename[self.idx2fileidx[idx]]}.npy'
                if not ecog_path.exists():
                    continue
            else:

                ecog_path = self.root_dir/self.paradigm/f'{self.ecog_fileprefix}{self.idx2fileidx[idx]}.npy'

            text_idx = self.fileidx2textidx[self.idx2fileidx[idx]]
            if text_idx not in data.keys():
                data[text_idx]=[]
            data[text_idx].append(ecog_path)
        trimmed_data={}
        
        for k,d in data.items():
            if len(d)>1:
                trimmed_data[k]=d
        data=trimmed_data
        return data
    
    
    
    def train_dataloader(self) -> DataLoader:
        
        data = self._load_data('train')
        dataset = ECoGDataset(data, load_list=self.load_list, text_list=self.text_list, transform=self.get_transform('train',**self.transform_config),
                                   only_hga=self.only_hga, )
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
    
    def val_dataloader(self) -> DataLoader:
        
        data = self._load_data('val')
        dataset = ECoGDataset(data, load_list=self.load_list, text_list=self.text_list ,transform=self.get_transform('train',**self.transform_config),only_hga=self.only_hga, )
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
    
    def test_dataloader(self) -> DataLoader:
        
        data = self._load_data('test')
        dataset = ECoGDataset(data, load_list=self.load_list, text_list=self.text_list, transform=self.get_transform('test',**self.transform_config),only_hga=self.only_hga, )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory,
            collate_fn=ECoGDataset.collate
        )
        return loader
    
    def get_transform(self,mode='train',
                     jitter_range=[0.8, 1.0],
                      jitter_max_start=400,
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
            transforms_=[]
            if 'cutdim' in transform_list:
                transforms_.append(CutDim(cutdim))
            if 'jitter' in transform_list and not no_jitter:
                transforms_.append(Jitter(jitter_range))    
            return transforms.Compose(transforms_)
      
    
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
    def __init__(self, fraction_range=[0.8,1.0],max_start=400):
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

    
