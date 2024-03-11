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

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
regex = re.compile('[%s]' % re.escape(string.punctuation))

LIBROSA_SILENCE_PAD=np.array([-9.5089, -9.4558, -9.4232, -9.4003, -9.4520, -9.5127, -9.4158, -9.5411,
        -9.6142, -9.5206, -9.4614, -9.4282, -9.5495, -9.4490, -9.3996, -9.5767,
        -9.4262, -9.5018, -9.6007, -9.5581, -9.6424, -9.6731, -9.5658, -9.4969,
        -9.3314, -9.4214, -9.5220, -9.5737, -9.4606, -9.6352, -9.6202, -9.6113,
        -9.6326, -9.5743, -9.5207, -9.3901, -9.4029, -9.4234, -9.3812, -9.4492,
        -9.5248, -9.5878, -9.5449, -9.4718, -9.4219, -9.4884, -9.5739, -9.3787,
        -9.4187, -9.3852, -9.3747, -9.4197, -9.5124, -9.5357, -9.4239, -9.4847,
        -9.4871, -9.4354, -9.4524, -9.4314, -9.4296, -9.4682, -9.3047, -9.4331,
        -9.4564, -9.4319, -9.4631, -9.4979, -9.4187, -9.4906, -9.4452, -9.4585,
        -9.4403, -9.4528, -9.4481, -9.4803, -9.4168, -9.3716, -9.4273, -9.4730])[None,:]
        
def butter_bandpass(cut, fs, order=5):
    if isinstance(cut,list) and len(cut) == 2:
        return butter(order, cut, fs=fs, btype='bandpass')
    else:
        return butter(order, cut, fs=fs, btype='low')

def butter_bandpass_filter(data, cut, fs, order=5):
    b, a = butter_bandpass(cut, fs, order=order)
    y = filtfilt(b, a, data,axis=0)
    return y

def buffer_array(arr,fb=10,eb=10):
    # arr: (L, d)
    fb = int(np.round(fb))
    eb = int(np.round(eb))
    if len(arr.shape)==1:
        # array is unit array
        return buffer_unit(arr, fb, eb, -1)
    else:
        if arr.shape[-1] ==80:
            arr = np.concatenate([np.zeros((fb,arr.shape[1]))+LIBROSA_SILENCE_PAD,
                              arr,
                              np.zeros((eb,arr.shape[1]))+LIBROSA_SILENCE_PAD])
        else:
            arr = np.concatenate([np.zeros((fb,arr.shape[1])),
                              arr,
                              np.zeros((eb,arr.shape[1]))])
        return arr

def buffer_unit(arr,fb=10,eb=10,pad_val=-1):
    # arr: (L,)
    dtype = arr.dtype
    arr = np.concatenate([np.zeros(fb)+pad_val,
                          arr,
                          np.zeros(eb)+pad_val])
    arr = arr.astype(dtype)
    return arr



def align(arr, aln, fb=10,eb=10, delay=0,factor=1):
    aln = aln*factor
    fb = fb*factor
    eb = eb*factor
    delay = delay*factor
    aln = aln+fb
    arr = buffer_array(arr,fb+delay,eb)
    return arr[aln.round().astype(int)]
    

class ECoGDataset(Dataset):
    
    def __init__(self, data, load_list=[], text_list=[], transform=None,only_hga=False,
                multi_hb_dirs=None, do_align=False, fb=10,eb=10, align_target=[], align_factors={}, delay=0):
        super().__init__()
        self.data = data
        self.transform = transform
        self.decimate=decimate
        self.load_list = load_list
        self.text_list = text_list
        self.only_hga = only_hga
        self.multi_hb_dirs = multi_hb_dirs
        self.do_align = do_align
        if self.do_align:
            self.multi_hb_dirs=None
        self.fb = fb
        self.eb = eb
        self.delay = delay
        self.align_target = align_target
        self.align_factors=align_factors
        for target in self.align_target:
            if target not in self.align_factors:
                self.align_factors[target]=1
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self,i):
        data = self.data[i]
        output = {}
        if self.do_align:
            aln = np.load(data['alignment'])
        else:
            aln = None
        for entry in self.load_list:
            if entry =='ecog':
                ecog = np.load(data['ecog'])
                if self.transform is not None and not self.do_align:
                    assert ecog is not None
                    ecog = self.transform(ecog)
                ecog = torch.from_numpy(ecog).float()
                if self.only_hga:
                    ecog = ecog[:,:253]
                output['ecog'] = ecog
                output['ecog_len'] = len(ecog)
            elif entry =='feature':
                if self.multi_hb_dirs is not None:
                    feature = np.load(random.choice(self.multi_hb_dirs)/Path(data['unit']).name)
                else:
                    feature = np.load(data['feature'])
                if entry in self.align_target:
                    feature = align(feature, aln,self.fb,self.eb,self.delay,factor=self.align_factors[entry])
                feature = torch.from_numpy(feature).float()
                output['feature'] =feature
                output['feature_len'] = len(feature)
            elif entry =='ema':
                feature = np.load(data['ema'])
                assert (~np.isnan(feature)).all()
                if entry in self.align_target:
                    feature = align(feature, aln,self.fb,self.eb,self.delay,factor=self.align_factors[entry])
                feature = torch.from_numpy(feature).float()
                
                output['ema'] =feature
                output['ema_len'] = len(feature)
                
            elif entry =='unitarr':
                feature = np.load(data['unitarr'])
                if entry in self.align_target:
                    feature = align(feature, aln,self.fb,self.eb,self.delay,factor=self.align_factors[entry])
                feature = torch.from_numpy(feature).long()
                output['unitarr'] =feature
                output['unitarr_len'] = len(feature)
                
            elif 'audio' in entry:
                wav,sr = sf.read(data[entry])
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                output['wav'] = wav
            elif entry == 'block':
                output['block'] = data[entry]
            else:
                raise NotImplementedError
        
        
        if 'unit' in data.keys() and self.multi_hb_dirs is not None:
            hb_tokens = np.load(random.choice(self.multi_hb_dirs)/data['unit'])
            text = ' '.join([str(t) for t in hb_tokens])
            data['unit'] = text
        
        if self.do_align and self.transform is not None:
            output = self.transform(output)
            for entry in self.align_target+['ecog']:
                output[f'{entry}_len'] = len(output[entry])
        if len(self.text_list) == 1:
            output['text'] = data[ self.text_list[0]]
        else:
            output['text'] = {entry:data[entry] for entry in self.text_list}
                
        return output
    
    @staticmethod
    def collate(batch):
        data = {}
        if 'ecog' in batch[0].keys():
            data['ecogs'] = nn.utils.rnn.pad_sequence([d['ecog'] for d in batch], batch_first=True, padding_value=0.0)
            data['ecog_lens'] = torch.tensor([d['ecog_len'] for d in batch])
            
        if 'feature' in batch[0].keys():
            data['features'] = nn.utils.rnn.pad_sequence([d['feature'] for d in batch], batch_first=True, padding_value=0.0)
            data['feature_lens'] = torch.tensor([d['feature_len'] for d in batch])
            
        if 'ema' in batch[0].keys():
            data['emas'] = nn.utils.rnn.pad_sequence([d['ema'] for d in batch], batch_first=True, padding_value=0.0)
            data['ema_lens'] = torch.tensor([d['ema_len'] for d in batch])
            
        if 'unitarr' in batch[0].keys():
            data['unitarrs'] = nn.utils.rnn.pad_sequence([d['unitarr'] for d in batch], batch_first=True, padding_value=-1)
            data['unitarr_lens'] = torch.tensor([d['unitarr_len'] for d in batch])
        
        if 'text' in batch[0].keys():
            if isinstance(batch[0]['text'], dict):
                data['texts']={entry: [d['text'][entry] for d in batch] for entry in batch[0]['text'].keys()}
            else:
                data['texts']=[d['text'] for d in batch]
        
        if 'block' in batch[0].keys():
            data['blocks'] = torch.tensor([d['block'] for d in batch], dtype=torch.long)
        if 'wav' in batch[0].keys():
            wav_input = processor([d['wav'] for d in batch],
                                       sampling_rate=16000, return_tensors="pt",
                                       padding=True)

            data['wavs'] = wav_input.input_values.detach()
            data['wav_lens'] = np.array([len(d['wav']) for d in batch])
        return data
    
class ECoGDataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir,
                 hb_dir,
                 ft_dir=None,
                 ema_dir=None,
                 aln_dir=None,
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
                 include_ema=False,
                 include_unitarr=False,
                 include_alignment=False,
                 shuffle_trainval_split=False,
                 train_files = None,
                 test_files = None,
                 val_files = None,
                 load_list=[],
                 text_list=[],
                 no_transform=False,
                 no_val_transform=False,
                 use_test_for_valid=False,
                 block_idxs=None,
                 only_hga=False,
                 relabeled = False,
                 bn2date= '/data/cheoljun/b3_misc/bn2date.npy',
                 convert_bn2date=False,
                 text_label_file=None,
                 ft_paradigm=None,
                 hb_paradigm=None,
                 audio_paradigm=None,
                 multi_hb_dirs=None,
                 do_align=False,
                 fb=10,
                 eb=10,
                 delay=0,
                 align_target=[],
                 align_factors={},
                 factor=4,
                 **kwargs,
                 ):
        super().__init__()
        
        
        if textidx_file is None:
            if text_label_file is not None:
                with open(text_label_file, 'r') as f:
                    texts = [t.rstrip() for t in f.readlines()]
                self.fileidx2textidx ={}
                textidx_file = None
            else:
                textidx_file = self.audio_dir/'file2textindex.npy'
                self.fileidx2textidx = np.load(textidx_file, allow_pickle=True)[()]
                texts = None
        else:
            self.fileidx2textidx = np.load(textidx_file, allow_pickle=True)[()]
        
        
        if ft_paradigm is None:
            ft_paradigm = paradigm.split('_recent')[0]
        if hb_paradigm is None:
            hb_paradigm = paradigm.split('_recent')[0]
        if audio_paradigm is None:
            audio_paradigm = paradigm.split('_recent')[0]    
        
        self.ft_dir = Path(ft_dir)/ft_paradigm if ft_dir is not None else None
        self.ema_dir = Path(ema_dir)/hb_paradigm if ema_dir is not None else None
        self.aln_dir = Path(aln_dir)/paradigm if aln_dir is not None else None
        self.hb_dir = Path(hb_dir)/hb_paradigm
        self.audio_dir = Path(audio_dir)/audio_paradigm
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
            if textidx_file is None:
                self.fileidx2textidx[fileidx] = texts.index(text)
        
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
        self.include_ema = include_ema
        self.include_unitarr = include_unitarr
        self.include_alignment = include_alignment
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
        if multi_hb_dirs is not None:
            self.multi_hb_dirs= [Path(d)/hb_paradigm for d in multi_hb_dirs]
        else:
            self.multi_hb_dirs = None
        self.no_val_transform=no_val_transform
        self.do_align=do_align
        self.fb=fb
        self.eb=eb
        self.delay=delay
        self.align_target=align_target
        self.factor = factor
        self.align_factors=dict(align_factors)
        global LIBROSA_SILENCE_PAD
        if not ((ft_dir is not None and 'librosa' in ft_dir) or (ema_dir is not None and 'librosa' in ema_dir)):
            LIBROSA_SILENCE_PAD = np.zeros((1,80))
        
    def _load_data(self, split):
        
        data_files = {'train':self.train_files,
                      'val':self.val_files,
                      'test':self.test_files}[split]
        data = []
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        for data_file in data_files:
            idx = '_'.join(data_file.split('_')[1:3])
            d_ = {} #[self.root_dir/self.paradigm/f'{self.ecog_fileprefix}{self.idx2fileidx[idx]}.npy', text]
            
            if self.include_ecog:
                if self.relabeled:
                    d_['ecog'] = self.root_dir/self.paradigm/f'{self.idx2filename[self.idx2fileidx[idx]]}.npy'
                    if not d_['ecog'].exists():
                        p = d_['ecog']
                        print(f'{str(p)} doesn\'t exist')
                        continue
                else:
                    
                    d_['ecog'] = self.root_dir/self.paradigm/f'{self.ecog_fileprefix}{self.idx2fileidx[idx]}.npy'
            
            if self.include_feature:
                d_['feature'] = self.ft_dir/f'{self.fileidx2textidx[self.idx2fileidx[idx]]}.npy'
             
            
            if self.include_unit:
                if self.multi_hb_dirs is not None:
                    d_['unit'] = f'{self.fileidx2textidx[self.idx2fileidx[idx]]}.npy'
                else:
                    hb_tokens = np.load(self.hb_dir/f'{self.fileidx2textidx[self.idx2fileidx[idx]]}.npy')
                    text = ' '.join([str(t) for t in hb_tokens])
                    d_['unit'] = text
                    
            if self.include_unitarr:
                d_['unitarr'] = self.hb_dir/f'{self.fileidx2textidx[self.idx2fileidx[idx]]}.npy'
            
            if self.include_alignment:
                d_['alignment'] = self.aln_dir/f'{self.idx2filename[self.idx2fileidx[idx]]}.npy'
                
            if self.include_ema:
                d_['ema'] = self.ema_dir/f'{self.fileidx2textidx[self.idx2fileidx[idx]]}.npy'
            
            if self.include_audio:
                d_['audio'] = self.audio_dir/f'{self.fileidx2textidx[self.idx2fileidx[idx]]}.wav'
            
            if self.include_phoneme:
                text= self.idx2text[idx]
                text = regex.sub('', text.lower())
                text = ' '.join([t for t in text.split(' ') if t != ''])
                d_['phoneme'] = text
            
            if self.include_block:
                block = int(idx.split('_')[0][1:])
                if self.convert_bn2date:
                    block = self.bn2date[block]
                block_idx = self.block_idxs[block]
                d_['block'] = block_idx
            data.append(d_)
        print(f'@@@@@@@@@ {split} - {len(data)} are found @@@@@@@@@@')
        return data
    
    
    
    def train_dataloader(self) -> DataLoader:
        
        data = self._load_data('train')
        dataset = ECoGDataset(data, load_list=self.load_list, text_list=self.text_list, transform=self.get_transform('train',**self.transform_config),
                                   only_hga=self.only_hga, multi_hb_dirs=self.multi_hb_dirs,
                             do_align=self.do_align, fb=self.fb, eb=self.eb, align_target=self.align_target,
                              delay=self.delay,align_factors=self.align_factors)
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
        transform = None if self.no_val_transform else self.get_transform('train',**self.transform_config)
        data = self._load_data('val')
        dataset = ECoGDataset(data, load_list=self.load_list, text_list=self.text_list ,
                              transform=transform,only_hga=self.only_hga, multi_hb_dirs=self.multi_hb_dirs,
                             do_align=self.do_align, fb=self.fb, eb=self.eb, align_target=self.align_target,
                              delay=self.delay, align_factors=self.align_factors)
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False, #self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=ECoGDataset.collate
        )
        return loader
    
    def test_dataloader(self) -> DataLoader:
        transform = None 
        data = self._load_data('test')
        dataset = ECoGDataset(data, load_list=self.load_list, text_list=self.text_list ,transform=transform,only_hga=self.only_hga, 
                              multi_hb_dirs=self.multi_hb_dirs,
                             do_align=self.do_align, fb=self.fb, eb=self.eb, align_target=self.align_target,delay=self.delay,
                             align_factors=self.align_factors)
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
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
            if 'cutdim' in transform_list and not self.do_align:
                transforms_.append(CutDim(cutdim))
            if 'jitter' in transform_list and not no_jitter:
                if self.do_align:
                    transforms_.append(MultiJitter(jitter_range,jitter_max_start,
                                                   targets=self.align_target, factor=self.factor))
                else:
                    transforms_.append(Jitter(jitter_range,jitter_max_start))
            if 'channeldrop' in transform_list and not self.do_align:
                transforms_.append(ChannelDrop(channeldropout_prob, channeldropout_rate))
            if 'scale' in transform_list and not self.do_align:
                transforms_.append(ScaleAugment(scaleaugmnet_range))
            if 'sample_window' in transform_list and not self.do_align:
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
    
class MultiJitter(object):
    def __init__(self, fraction_range=[0.8,1.0],max_start=100, targets=[],factor=4):
        self.max_start = 400
        self.targets=targets
        self.fraction_pool = np.linspace(fraction_range[0],fraction_range[1],5)
        self.factor = factor
        
    def __call__(self, d):
        x = d['ecog']
        fraction = np.random.choice(self.fraction_pool, 1)[0]
        start_f = np.random.uniform()*(1-fraction)
        end_f = start_f+fraction
        si,ei = int(len(x)*start_f),max(len(x),len(x)*end_f)
        si = min(self.max_start,si)
        d['ecog']=x[si:ei]
        for target in self.targets:
            d[target] = d[target][si//self.factor:ei//self.factor]
        return d
    
    
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

    
