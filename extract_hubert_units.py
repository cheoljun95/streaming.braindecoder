import torch
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import tqdm
import argparse
from model.tokenizer import HuBERTTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--rank",type=int, default=0, 
                    help="Index of process in parallel processing.")
parser.add_argument("--n",type=int, default=1, 
                    help="Total number of parallel processing.")
parser.add_argument("--km_n", type=int, default=100, 
                    help="Number of clusters used for hubert units. All modules are based on 100 cluster size.")
parser.add_argument("--data_path", type=str, 
                    help="Directory with audio files to extract hubert units. All audio should be wav files.")
parser.add_argument("--save_path", type=str, 
                    help="Saving destination.")
parser.add_argument("--dedup", action="store_true", 
                    help="Indicate whether to deduplicate units. e.g., [71 71 71 52 52 1] --> [71 52 1]")
parser.add_argument("--device", type=str, default="cuda:0",
                   help="Device for running model. default: cuda:0. If no gpu available, set as \"cpu\".")

if __name__=='__main__':
    args = parser.parse_args()
    source_dir = Path(args.data_path)
    save_dir = Path(args.save_path)
    save_dir.mkdir(exist_ok=True)
    device = args.device
    km_n = args.km_n
    wav_files = [f for f in source_dir.glob("*.wav")]
    wav_files.sort()
    chunk_len = int(len(wav_files)/args.n)+1
    wav_files = wav_files[chunk_len*args.rank:chunk_len*(args.rank+1)]

    tokenizer= HuBERTTokenizer(pre_tokenized=False, km_n=km_n, 
                               device=device, collapse=args.dedup)
    
    for wav_file in tqdm.tqdm(wav_files):
        file_name = save_dir/f'{wav_file.stem}.npy'
        if file_name.exists():
            continue
        wav,sr = sf.read(wav_file)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        tokens = tokenizer.tokenize(wav)
        np.save(file_name, tokens)
    
    
        