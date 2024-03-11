import torch
from pathlib import Path
import numpy as np
import soundfile as sf
import tqdm
import pickle
import argparse
import sys
import librosa

sys.path.append('..')
from model.tokenizer import HuBERTTokenizer
from scipy.signal import butter, lfilter, filtfilt

def butter_bandpass(cut, fs, order=5):
    
    if isinstance(cut,list) and len(cut) == 2:
        return butter(order, cut, fs=fs, btype='bandpass')
    else:
        return butter(order, cut, fs=fs, btype='low')

def butter_bandpass_filter(data, cut, fs, order=5):
    b, a = butter_bandpass(cut, fs, order=order)
    y = filtfilt(b, a, data,axis=0)
    return y

parser = argparse.ArgumentParser()
parser.add_argument("--rank",type=int,default=0)
parser.add_argument("--n",type=int,default=1)
parser.add_argument("--km_n", type=int, default=100)
parser.add_argument("--data_root",type=str,default='/data/cheoljun/b3_audio_scale-2')
parser.add_argument("--save_path",type=str, default=None)
parser.add_argument("--paradigm",type=str, default='tm1k')

if __name__=='__main__':
    args = parser.parse_args()
    source_dir = Path(args.data_root)
    if args.save_path is None:
        save_dir = source_dir/'unit_label'
        save_dir.mkdir(exist_ok=True)
    else:
        save_dir = Path(args.save_path)
        
    device='cuda'
    km_n = args.km_n
    save_dir = save_dir/f'hubert-l6_km{km_n}'
    save_dir.mkdir(exist_ok=True)
    wav_files = []
    for split in [args.paradigm]: #['tm1k']:
        wav_files += [f for f in (source_dir/split).glob('*.wav')]

    wav_files.sort()
    chunk_len = int(len(wav_files)/args.n)+1
    wav_files = wav_files[chunk_len*args.rank:chunk_len*(args.rank+1)]

    tokenizer= HuBERTTokenizer(pre_tokenized=False, km_n=km_n, device=device)
    for wav_file in tqdm.tqdm(wav_files):
        parent = wav_file.parent.stem
        (save_dir/parent).mkdir(exist_ok=True)
        file_name = save_dir/parent/f'{wav_file.stem}.npy'
        if file_name.exists():
            continue
        wav,sr = sf.read(wav_file)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        tokens = tokenizer.tokenize(wav)
        
        
        np.save(file_name, tokens)
    
    
        