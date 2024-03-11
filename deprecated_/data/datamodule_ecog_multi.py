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
        self.input_sr = 200
        self.decimate = 4
        
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def _preprocess(hga,q=4):
        # hga: (T, d=506)
        # low-pass filter and down-sample
        hga_lp = butter_bandpass_filter(hga,100/q,200)
        hga_lpds = decimate(hga_lp, q, axis=0)
        hga_lpds = normalize(hga_lpds,0)
        return hga_lpds
    
    def __getitem__(self,i):
        ecog_path, text, ema_path, segment = self.data[i]
        start,end = max(0,segment[0]-0.5), segment[1]+0.5
        ema = np.load(ema_path)
        ecog = np.load(ecog_path)
        if self.do_preprocess:
            ecog = self._preprocess(ecog, self.decimate)
        if self.transform is not None:
            ecog = self.transform(ecog)
        ecog = torch.from_numpy(ecog.copy()).float()
        ema = torch.from_numpy(ema).float()
        segment = torch.zeros(len(ecog))
        sr = self.input_sr/self.decimate
        segment[int(sr*start):int(sr*end)] = 1
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
                  'text':text,
                  'ema':ema,
                  'ema_len': len(ema),
                  'segment': segment}
        
        return output
    
    @staticmethod
    def collate(batch):
        data = {}
        data['ecogs'] = nn.utils.rnn.pad_sequence([d['ecog'] for d in batch], batch_first=True, padding_value=0.0)
        data['ecog_lens'] = torch.tensor([d['ecog_len'] for d in batch])
        data['phonemes'] = nn.utils.rnn.pad_sequence([d['phoneme'] for d in batch], batch_first=True, padding_value=-1)
        data['phoneme_lens'] = torch.tensor([d['phoneme_len'] for d in batch])
        data['texts']=[d['text'] for d in batch]
        data['emas'] = nn.utils.rnn.pad_sequence([d['ema'] for d in batch], batch_first=True, padding_value=0.0)
        data['ema_lens'] = torch.tensor([d['ema_len'] for d in batch])
        data['segments'] = nn.utils.rnn.pad_sequence([d['segment'] for d in batch], batch_first=True, padding_value=-1)
        return data
    
class ECoGPhonemeDataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir,
                 segment_file = '/data/cheoljun/b3_features/tm1k_segment.npy',
                 textidx_file='/data/cheoljun/b3_audio_scale-2/tm1k/file2textindex.npy',
                 ema_dir = '/data/cheoljun/b3_ema_scale-2/tm1k',
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
                 ):
        super().__init__()
        
        
        self.fileidx2segment = np.load(segment_file, allow_pickle=True)[()]
        self.fileidx2textidx = np.load(textidx_file, allow_pickle=True)[()]
        self.ema_dir = Path(ema_dir)
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
        for fileidx, (idx, text) in enumerate(zip(df['gt_idx'].values, df['gt_text'].values)):
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
        
    def _load_data(self, split):
        
        data_files = {'train':self.train_files,
                      'val':self.val_files,
                      'test':self.test_files}[split]
        data = []
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        for data_file in data_files:
            idx = int(data_file.split('_')[-1].split('.')[0])
            text= self.idx2text[idx]
            text = regex.sub('', text.lower())
            text = ' '.join([t for t in text.split(' ') if t != ''])
            fileidx=self.idx2fileidx[idx]
            data.append([self.root_dir/self.paradigm/f'{self.ecog_fileprefix}{fileidx}.npy', text,
                         self.ema_dir/f'{self.fileidx2textidx[fileidx]}.npy', self.fileidx2segment[fileidx]])
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
                     blackout_len=0.30682868940865543,
                     blackout_prob =0.04787032280216536,
                     additive_noise_level= 0.0027354917297051813,
                     chan_noise=0.,
                     scale_augment_low= 0.9551356218945801,
                     scale_augment_high = 1.0713824626558794,
                     jitter_amt=1,
                      decimation=4,
                     ):
        
        if mode =='train':
            blackout = Blackout(blackout_len, blackout_prob)
            noise = AdditiveNoise(additive_noise_level)
            chan_noise = LevelChannelNoise(chan_noise)
            scale = ScaleAugment(scale_augment_low, scale_augment_high)
            '''
            jitter = Jitter((-1, 8), 
                            (0.5, 7.5),
                            jitter_amt=jitter_amt,
                            decimation=decimation)
            '''
            return transforms.Compose([blackout, noise,
                                                 chan_noise, scale])
        else:
            return None
            '''
            jitter = Jitter((-1, 8), 
                            (0.5, 7.5),
                            jitter_amt=jitter_amt,
                            decimation=decimation)
            return transforms.Compose([jitter])
            '''
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
    
class Jitter(object):
    """
    randomly select the default window from the original window
    scale the amt of jitter by jitter amt
    validation: just return the default window. 
    """
    def __init__(self, original_window, default_window, jitter_amt, sr=200, decimation=6, validate=False):
        self.original_window = original_window
        self.default_window = default_window
        self.jitter_scale = jitter_amt
        
        default_samples = np.asarray(default_window) - self.original_window[0]
        default_samples = np.asarray(default_samples)*sr/decimation
        
        default_samples[0] = int(default_samples[0])
        default_samples[1] = int(default_samples[1])
        
        self.default_samples = default_samples
        self.validate = validate
        
        self.winsize = int(default_samples[1] - default_samples[0])+1
        self.max_start = int(int((original_window[1] - original_window[0])*sr/decimation) - self.winsize)
        
        
    def __call__(self, sample):
        if self.validate: 
            return sample[int(self.default_samples[0]):int(self.default_samples[1])+1, :]
        else: 
            start = np.random.randint(0, self.max_start)
            scaled_start = np.abs(start-self.default_samples[0])
            scaled_start = int(scaled_start*self.jitter_scale)
            scaled_start = int(scaled_start*np.sign(start-self.default_samples[0]) + self.default_samples[0])
            return sample[scaled_start:scaled_start+self.winsize]
        
        
class Blackout(object):
    """
    The blackout augmentation.
    """
    def __init__(self, blackout_max_length=0.3, blackout_prob=0.5):
        
        self.bomax = blackout_max_length
        self.bprob = blackout_prob
        
        
    def __call__(self, sample):
      
        blackout_times = int(np.random.uniform(0, 1)*sample.shape[0]*self.bomax)
        start = np.random.randint(0, sample.shape[0]-sample.shape[0]*self.bomax)
        if random.uniform(0, 1) < self.bprob: 
            sample[start:(start+blackout_times), :] = 0
        return sample
    
class ChannelBlackout(object):
    """
    Randomly blackout a channel. 
    """
    def __init__(self, blackout_chans_max=20, blackout_prob=0.2):
        self.bcm = blackout_chans_max
        self.bp = blackout_prob
    def __call__(self, sample):
        if random.uniform(0, 1) < self.bp:
            inds = np.arange(sample.shape[-1])
            np.random.shuffle(inds)
            boi = inds[:self.bcm]
            sample[:, bcm] = 0

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


class Normalize(object):
    def __init__(self, axis):
        """
        Does normalization func
        """
        self.axis = axis
        
    def __call__(self, sample):
        sample_ = normalize(sample, axis=self.axis)
        return sample_
    
class AdditiveNoise(object):
    def __init__(self, sigma):
        """
        Just adds white noise.
        """
        self.sigma = sigma
        
    def __call__(self, sample):
        sample_ = sample + self.sigma*np.random.randn(*sample.shape)
        return sample_
        
class ScaleAugment(object):
    def __init__(self, low_range, up_range):
        self.up_range = up_range # e.g. .8
        self.low_range = low_range
        print('scale', self.low_range, self.up_range)
#         assert self.up_range >= self.low_range
    def __call__(self, sample):
        multiplier = np.random.uniform(self.low_range, self.up_range)
        return sample*multiplier

class LevelChannelNoise(object):
    def __init__(self, sigma, channels=128):
        """
        Sigma: the noise std. 
        """
        self.sigma= sigma
        self.channels = channels
        
    def __call__(self, sample):
        sample += self.sigma*np.random.randn(1,sample.shape[-1]) # Add uniform noise across the whole channel. 
        return sample