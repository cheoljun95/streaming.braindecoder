import numpy as np
import torch
import torch.nn as nn
from model.ecog2speech_rnnt import MunitRNNT
from model.rnnt_beamsearch import RNNTBeamSearch
from synthesis.hificar_dur import HifiCarDurationSynthesizer
import time
import tqdm
from pathlib import Path

class ZeroCrossingCleaner:
    def __init__(self, chunk_size=320):
        self.chunk_size = chunk_size
        self.sign = None
        
    def clear(self):
        self.sign = None
        
    def __call__(self,wav):
        zc = (wav[1:]*wav[:-1])<0
        search_size=max(0,len(wav)-self.chunk_size)
        if self.sign is None:
            nearest_zc = search_size
        else:
            zc_sign = np.nonzero(zc &(np.diff(wav)*self.sign >0))[0]
            nearest_zc = zc_sign[abs(zc_sign-search_size).argmin()]
        last_zc = np.nonzero(zc)[0][-1]
        self.sign = (np.diff(wav)[last_zc]>0)*2.0-1
        
        wav= wav[nearest_zc:last_zc+1]
        return wav
    
class UnitStreamer(nn.Module):
    
    def __init__(self, preloaded_rnnt, vocoder_ckpt_path, device='cuda', audio_sr=16000,
                 num_prev_unit=16, expand_factor=3.0, beam_width=20, best_n=10, output_len=1280, 
                 use_buffer=False, emit_n=4,**kwargs):
        super().__init__()
        
        self.decoder = preloaded_rnnt
        self.vocoder = HifiCarDurationSynthesizer(model_ckpt = vocoder_ckpt_path,
                                                 device= device,
                                                 output_sr = audio_sr,
                                                 num_prev_unit = num_prev_unit,
                                                 use_buffer=use_buffer, emit_n=emit_n)
        
        self.expand_factor = expand_factor
        
        #self.cleaner = ZeroCrossingCleaner(output_len)
        self.device = device
        self.beam_width = beam_width
        self.best_n = best_n
        self.output_len = output_len
        
        self.wav_buffer = np.array([])
        self.state_ = None
        self.hypos_ = None
        self.output_name = 'wav'
        self.unit_history = []
        self.latency_history = []
        self.latency_history_names = ["beam_search", "synthesis"]
        
    def clear_cache(self):
        self.wav_buffer = np.array([])
        self.unit_buffer = np.array([])
        self.state_ = None
        self.hypos_ = None
        self.vocoder.reset()
        self.unit_history = []
        self.latency_history = []
    
    def output_wav_from_buffer(self):
        if len(self.wav_buffer) ==0:
            return np.zeros(self.output_len)
        else:
            output = self.wav_buffer[:self.output_len]
            self.wav_buffer = self.wav_buffer[self.output_len:]
            if len(output) < self.output_len:
                output = np.concatenate([output, np.zeros(self.output_len-len(output))])
            return output
        
    def push_wav_buffer(self, wav):
        self.wav_buffer = np.concatenate([self.wav_buffer, wav])
        
    def forward(self, x):
        unit_tensor = None
        times = []
        if x is not None:
            start = time.time()
            with torch.no_grad():
                hypos, state = self.decoder.infer(x, torch.tensor([x.shape[-2]], device=self.device), self.beam_width,
                                                  state=self.state_, hypothesis=self.hypos_)
            
            if self.hypos_ is not None:
                units = hypos[0][0][len(self.hypos_[0][0]):]
            else:
                units = hypos[0][0][1:]
            self.unit_history.append(units)
            self.hypos_ = hypos[:self.best_n]
            self.state_ = state
            if len(units)>0:
                unit_tensor = torch.tensor(units).long().to(self.device).unique_consecutive()
            times.append(time.time()-start)
        else:
            times.append(0)
        start = time.time() 
        wav = self.vocoder.synthesize_v2(unit_tensor,alpha=self.expand_factor,min_dur=1)
        if wav is not None:
            self.push_wav_buffer(wav)
        times.append(time.time()-start)
        self.latency_history.append(times)
        return self.output_wav_from_buffer()
        
        
        
class TextStreamer(nn.Module):
    
    def __init__(self, preloaded_rnnt, preloaded_tokendecoder,  beam_width=20,
                 best_n=10, device='cuda',full_hypo=False, output_type='text', **kwargs):
        super().__init__()
        
        self.decoder=preloaded_rnnt
        self.token_decoder = preloaded_tokendecoder
        
        self.device = device
        self.beam_width = beam_width
        self.best_n = best_n
        
        self.state_ = None
        self.hypos_ = None
        self.text = ''
        self.full_hypo = full_hypo
        self.output_name = output_type
        self.latency_history = []
        self.latency_history_names = ["beam_search"]
        
    def clear_cache(self):
        self.state_ = None
        self.hypos_ = None
        self.text = ''
        self.latency_history = []
    
    def forward(self, x):
        times = []
        if x is not None:
            start = time.time()
            with torch.no_grad():
                hypos, state = self.decoder.infer(x, torch.tensor([x.shape[-2]]), self.beam_width, state=self.state_,
                                             hypothesis=self.hypos_)

            self.hypos_ = hypos[:self.best_n]
            self.state_ = state

            output_hypo_len = len(hypos) if self.full_hypo else 1
            texts = []
            
            for hypo in hypos[:output_hypo_len]:
                units = hypo[0][1:]

                text = self.token_decoder(units)
            
                texts.append([text,hypo[3]])
            if output_hypo_len == 1:
                texts = texts[0][0]
            self.text = texts
            times.append(time.time()-start)
        else:
            times.append(0)
        self.latency_history.append(times)
        return self.text
        
class MultiStreamer(nn.Module):
    
    def __init__(self, rnnt_ckpt_path, streamer_configs, context_buffer=0, device='cuda', buffer_size=16, **kwargs):
        
        super().__init__()
        if torch.cuda.is_available() and 'cuda' in device:
            device = device
        else:
            print("no gpu available")
            device = "cpu"
        rnnt_ckpt = torch.load(rnnt_ckpt_path,map_location='cpu')
        
        model = MunitRNNT(**rnnt_ckpt['config'])
        
        model.load_state_dict(rnnt_ckpt['state_dict'],strict=False)
        model = model.eval().to(device)
        
        self.streamers = nn.ModuleList()
        for module_i, streamer_config in streamer_configs.items():
            module_i = int(module_i)
            rnnt_ = RNNTBeamSearch(model.rnnts[module_i], model.blank_ids[module_i],
                                   step_max_tokens=streamer_config['max_tokens'],temperature=streamer_config['temperature'])
            
            if streamer_config['type'] == 'unit':
                self.streamers.append(UnitStreamer(preloaded_rnnt=rnnt_,device=device,
                                                   **streamer_config))
            else:
                self.streamers.append(TextStreamer(preloaded_rnnt=rnnt_,
                                                   preloaded_tokendecoder=model.token_decoders[module_i],
                                                   device=device,
                                                   output_type=streamer_config['type'], **streamer_config))
                
        self.context_buffer = context_buffer
        self.device = device
        self.ftextractor  = model.feature_extractor
        self.buffer_size = buffer_size
        self.context_buffer_ = None
        self.ft_state_ = None
        self.st_state_ = None
        self.latency_history = []
        self.latency_history_names = ["transcriber"]
        
    def clear_cache(self):
        for streamer_ in self.streamers:
            streamer_.clear_cache()
        self.context_buffer_ = None
        self.ft_state_ = None
        self.st_state_ = None
        self.latency_history = []

    def forward(self, x):
        times = []
        start = time.time()
        if x is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device =self.device, dtype=torch.float32)

            x = x.to(self.device)    

            if self.context_buffer >0:
                if self.context_buffer_ is None:
                    self.context_buffer_ = torch.zeros(self.context_buffer, x.shape[1], device=self.device, dtype=torch.float32)
                x = torch.cat([self.context_buffer_, x],0)
                if len(x) > self.context_buffer:
                    self.context_buffer_ = x[-self.context_buffer:]
                elif len(x) == self.context_buffer:
                    self.context_buffer_ = x
                else:
                    self.context_buffer_ = torch.cat([self.context_buffer_[-(self.context_buffer-len(x)):],
                                                    x],0)
            if self.ftextractor is not None:
                with torch.no_grad():
                    x,_,  self.ft_state_ =self.ftextractor.infer(x.unsqueeze(0), torch.tensor([x.shape[1]],device=x.device),
                                                       self.ft_state_)
            times.append(time.time()-start)
        else:
            times.append(0)
        self.latency_history.append(times)
        outputs = {}
        for streamer_ in self.streamers:
            outputs[streamer_.output_name] = streamer_(x)
        outputs['time'] = time.time()-start
        return outputs
        
   
            
def build_dualstreamer_from_file(config_file_path, base_dir=None, **kwargs):
    import yaml
    if not Path(config_file_path).exists():
        config_file_path = str(Path(base_dir)/config_file_path)
    cfg = yaml.load(open(config_file_path), Loader=yaml.FullLoader)
    # override the path
    for arg_name, arg_val in kwargs.items():
        if arg_name in cfg.keys():
            cfg[arg_name] = arg_val
    if base_dir is not None:
        def add_base_dir(cfg_, base_dir):
            if not isinstance(cfg_, dict):
                return cfg_
            for name, data in cfg_.items():
                if isinstance(name, str) and 'path' in name and data != None:
                    cfg_[name]=str(Path(base_dir)/data)
                else:
                    cfg_[name] = add_base_dir(data, base_dir)
            return cfg_
        cfg = add_base_dir(cfg,base_dir)
        
    for key, val in kwargs.items():
        cfg[key] = val
            
    multistreamer = MultiStreamer(**cfg)
    return multistreamer, cfg


