import torch
from pathlib import Path
from streaming.streamer import MultiStreamer
import hydra
import torchaudio
import tqdm, string, re
import numpy as np
import soundfile as sf
from torchaudio.functional import edit_distance
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from hydra.utils import get_original_cwd

regex = re.compile('[%s]' % re.escape(string.punctuation))

def get_asr(wavs, processor, asr):
    if wavs is None or len(wavs)==0:
        return ''
    with torch.no_grad():
        input_features = processor(wavs, sampling_rate=16000, return_tensors="pt").input_features.to('cuda') 
        predicted_ids = asr.generate(input_features)
        synth_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return synth_text

def flatten_list(data):
    result = []
    for i in data:
        result.extend(i)
    return result

def parse_data(files, data_dir):
    data = []
    with open(files, "r") as f:
        for line in f.readlines():
            filename, text, hb_unit = line.split("|")
            text = regex.sub('', text.lower())
            text = ' '.join([t for t in text.split(' ') if t != ''])
            data.append({"ecog": data_dir/filename,
                         "phoneme": text,
                         "unit": hb_unit})
    return data
        

def to_text(arr):
    new_arr =[]
    for x in arr:
        x =' '.join([str(v) for v in x])
        new_arr.append(x)
    return new_arr

def fix_path(path):
    if str(path)[0] !='/':
        return Path(get_original_cwd())/path
    else:
        return path

@hydra.main(config_path='eval_configs')
def main(cfg):
    device = cfg['device']
    if "cuda" in device and not torch.cuda.is_available():
        print("Device is set as cuda but cuda is not available. Running on CPU instead.")
        device = "cpu"
        cfg['device'] = "cpu"
    # set up saving directory
    save_path =fix_path(cfg['save_path'])
    save_path.mkdir(exist_ok=True)
    metric_paths = {}
    for metric, metric_path in cfg['metrics'].items():
        metric_paths[metric] = save_path/metric_path
        metric_paths[metric].mkdir(exist_ok=True)
    
    # load streamer
    cfg['rnnt_ckpt_path'] = str(fix_path(cfg['rnnt_ckpt_path']))
    try:
        cfg['streamer_configs'][0]['vocoder_ckpt_path'] = str(fix_path(cfg['streamer_configs'][0]['vocoder_ckpt_path']))
    except:
        pass
    multistreamer = MultiStreamer(**cfg)
    
    # load asr model and processor
    processor = WhisperProcessor.from_pretrained(cfg['asr_model'])
    asr = WhisperForConditionalGeneration.from_pretrained(cfg['asr_model'])
    asr = asr.to(device)
    
    # load dataset
    test_files = fix_path(cfg['test_files'])
    data_dir = fix_path(cfg['data_dir'])
    test_data = parse_data(test_files, data_dir)
    buffer_size = cfg['buffer_size']
    is_chance = cfg['is_chance']
    
    for di in tqdm.tqdm(range(len(test_data))):
        
        x = test_data[di]
        input_all = torch.from_numpy(np.load(x['ecog'])).float().to(device)
        if is_chance:
            idxs = np.arange(input_all.shape[-1])
            np.random.shuffle(idxs)
            input_all = input_all[..., idxs]
        np.save(metric_paths['gt_unit']/f'{di}.npy', x['unit'])
        np.save(metric_paths['gt_text']/f'{di}.npy', x['phoneme'])
        
        multistreamer.clear_cache()
        wavs = []
        anchors = list(range(0,int(len(input_all)//buffer_size*buffer_size),buffer_size))
        if len(anchors) ==0:
            anchors = [0]
        text_history = []
        unit_history = []
        for i in anchors:
            ecog = input_all[i:i+buffer_size]
            outputs = multistreamer(ecog)
            wav= outputs['wav']
            text=outputs['text']
            text_history.append(text)
            unit_history.append(flatten_list(multistreamer.streamers[0].unit_history))
            wavs.append(wav)
        
        # clearing buffer
        for i in range(100):
            wavs.append(multistreamer(None)['wav'])
        wavs = np.concatenate(wavs)
        synth_text = get_asr(wavs, processor, asr)
        try:
            sf.write(metric_paths["pred_audio"]/f'{di}.wav', wavs, 16000)
            np.save(metric_paths["asr_transcript"]/f'{di}.npy', get_asr(wavs, processor, asr))
        except:
            sf.write(metric_paths["pred_audio"]/f'{di}.wav', np.zeros(8000), 16000)
            np.save(metric_paths["asr_transcript"]/f'{di}.npy', "")
        np.save(metric_paths["pred_text"]/f'{di}.npy', text)
        units = flatten_list(multistreamer.streamers[0].unit_history)
        np.save(metric_paths["pred_text"]/f'{di}.npy', units )
        np.save(metric_paths["speech_emission"]/f'{di}.npy', to_text(unit_history))
        np.save(metric_paths["text_emission"]/f'{di}.npy', text_history)
    
if __name__ =='__main__':
    main()