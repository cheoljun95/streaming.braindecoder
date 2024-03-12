import torch.nn as nn
import torchaudio
import torch
import string
import numpy as np
try:
    import soundfile as sf
    import librosa
except:
    librosa = None
    sf = None


class GraphemeTokenizer(nn.Module):
    def __init__(self,include_space=True, **kwargs):
        super().__init__()
        self.charactors = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
                           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
                           'u', 'v', 'w', 'x', 'y', 'z',]
        if include_space:
            self.charactors.append(' ')
        self.ch2idx = {ch:i for i, ch in enumerate(self.charactors)}
        self.pad_id = len(self.charactors) 
        self.blank =  len(self.charactors)+1
        
        
    def __call__(self, texts, **kwargs):
        texts=[text.lower() for text in texts]
        token_idxs = [[self.ch2idx[t] for t in text] for text in texts]
        token_idxs = [torch.tensor([self.blank] + idxs) for idxs in token_idxs]
        token_lengths = torch.tensor([len(idxs) for idxs in token_idxs])
        
        token_idxs = nn.utils.rnn.pad_sequence(token_idxs,batch_first=True, padding_value=self.pad_id)
        return token_idxs, token_lengths, texts
    
    def get_decoder(self):
        return lambda tokens: ''.join([self.charactors[i] for i in tokens])
    
    def get_remove_tokens(self):
        return [self.pad_id, self.blank]
    
    def pad_id(self):
        return self.pad_id
    
    def bos_id(self):
        return self.blank
    
    def eos_id(self):
        return self.blank
    
class PhonemeTokenizer(nn.Module):
    def __init__(self, include_stress=True,include_space=True,wrap_sent=False, **kwargs):
        super().__init__()
        from g2p_en import G2p
        self.g2p = G2p()
        self.phonemes = ['SIL', 'SPN', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
                         'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                         'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2',
                         'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
                         'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S',
                         'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
        self.include_space = include_space
        self.wrap_sent = wrap_sent
        
        if self.include_space:
            self.phonemes.append('|')
            
        
        self.include_stress = include_stress
        if not self.include_stress:
            phonemes_ = []
            for ph in self.phonemes:
                ph=self._remove_stressed_vowel(ph)
                if ph not in phonemes_:
                    phonemes_.append(ph)
            self.phonemes=phonemes_
        if '|' in self.phonemes:
            self.space_id = self.phonemes.index('|')
        if self.wrap_sent:
            self.phonemes += ['<BOS>', '<EOS>']
            self.bos_id = len(self.phonemes)-2
            self.eos_id = len(self.phonemes)-1
            
        self.ph2idx = {ph:i for i, ph in enumerate(self.phonemes)}
        self.pad_id = len(self.phonemes) 
        self.blank =  len(self.phonemes)+1
        self.vocab_size = len(self.phonemes)+2
        
    @staticmethod
    def _remove_stressed_vowel(ph):
        if ph[-1] in ['0','1','2']:
            ph = ph[:-1]
        return ph
    
    @staticmethod
    def _convert_space(ph):
        return '|' if ph ==' ' else ph
    
    def __call__(self, texts, **kwargs):
        texts = [self.g2p(text.lower()) for text in texts]
        if not self.include_stress:
            texts = [[self._remove_stressed_vowel(ph) for ph in text] for text in texts]
        texts = [[self._convert_space(ph) for ph in text] for text in texts]
        texts = [[ph for ph in text if ph in self.phonemes] for text in texts]
        token_idxs = [[self.ph2idx[t] for t in text] for text in texts]
        if self.wrap_sent:
            token_idxs = [[self.bos_id]+idxs+[self.eos_id] for idxs in token_idxs]
        token_idxs = [torch.tensor([self.blank] + idxs) for idxs in token_idxs]
        token_lengths = torch.tensor([len(idxs) for idxs in token_idxs])
        
        token_idxs = nn.utils.rnn.pad_sequence(token_idxs,batch_first=True, padding_value=self.pad_id)
        return token_idxs, token_lengths, [' '.join(text) for text in texts]
    
    def get_decoder(self):
        return lambda tokens: ' '.join([self.phonemes[i] for i in tokens])
    
    def get_remove_tokens(self):
        remove_tokens = [self.pad_id, self.blank, 0, 1]
        if self.wrap_sent:
            remove_tokens+=[self.bos_id, self.eos_id, 0, 1]
        return remove_tokens
    
    def pad_id(self):
        return self.pad_id
    
    def bos_id(self):
        return self.blank
    
    def eos_id(self):
        return self.blank

class BPETokenizer(nn.Module):
    
    def __init__(self,blank,**kwargs):
        super().__init__()
        self.blank=blank
        bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
        self.tokenizer = bundle.get_token_processor()
        self.vocab_size = 4096+2
        
    def __call__(self, texts, **kwargs):
        texts=[text.lower() for text in texts]
        token_idxs = self.tokenizer.sp_model.Encode(texts)
        token_idxs = [torch.tensor([self.blank] + idxs) for idxs in token_idxs]
        token_lengths = torch.tensor([len(idxs) for idxs in token_idxs])
        
        token_idxs = nn.utils.rnn.pad_sequence(token_idxs,batch_first=True, padding_value=self.tokenizer.sp_model.pad_id())
        return token_idxs, token_lengths, texts
    
    def get_decoder(self):
        return self.tokenizer.sp_model.DecodeIds
    
    def get_remove_tokens(self):
        return [self.tokenizer.sp_model.pad_id(),
                self.tokenizer.sp_model.bos_id(),
                self.tokenizer.sp_model.eos_id()]
    
    def pad_id(self):
        return self.tokenizer.sp_model.pad_id()
    
    def bos_id(self):
        return self.tokenizer.sp_model.bos_id()
    
    def eos_id(self):
        return self.tokenizer.sp_model.eos_id()
    
class HuBERTTokenizer(nn.Module):
    
    def __init__(self, pre_tokenized=True, km_n=100, device='cuda', collapse=True, spm=None, **kwargs):
        super().__init__()
        if not pre_tokenized:
            assert km_n in [50, 100, 200], "Only km_n in [50, 100, 200] is supported"
            from pathlib import Path
            module_path = Path(__file__)
            self.km_path = module_path.parent.parent/'km_models'
            if not self.km_path.exists():
                self.km_path.mkdir(exist_ok=True)
            self.layer = 6
            self.km_path = self.km_path/f'hubert-l{self.layer}-km{km_n}.bin'
            if not self.km_path.exists():
                import wget
                _ = wget.download(f'https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km{km_n}/km.bin', out=str(self.km_path))
            import joblib
            self.km_model = joblib.load(open(str(self.km_path), "rb"))
            self.km_model.verbose = False
            from transformers import Wav2Vec2Processor, HubertModel
            self.device = device
            self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
            self.speech_encoder = HubertModel.from_pretrained('facebook/hubert-base-ls960').to(self.device)
            
        if spm is not None:
            import sentencepiece
            from pathlib import Path
            if not Path(spm).exists():
                spm = str(Path(__file__).parent.parent/'spm'/f'hubert-l6_km{km_n}_{spm}.model')
            
            spm =str(spm)
            self.spm = sentencepiece.SentencePieceProcessor(model_file=spm)
            self.valid_unicode = np.load(Path(__file__).parent.parent/'misc'/'valid_unicode.npy')
            self.reverse = {code:i for i,code in enumerate(self.valid_unicode)}
            spm_file_tag=Path(spm).stem
            if 'bpe' in spm_file_tag:
                vocab_size = int(spm_file_tag.split('bpe')[-1])
            elif 'unigram' in spm_file_tag:
                vocab_size = int(spm_file_tag.split('unigram')[-1])
            else:
                raise NotImplemented
            self.pad_id = vocab_size
            self.blank = vocab_size+1
            self.collapse=True
            self.vocab_size = vocab_size+2
        else:
            self.spm = None 
            self.pad_id = km_n 
            self.blank = km_n+1
            self.collapse = collapse
            self.vocab_size = km_n+2
            
        self.pre_tokenized = pre_tokenized
        
    def get_feature(self, wav, sr=16000):
        assert not self.pre_tokenized, "set pre_tokenized=False to load quantizer model"
        inputs=self.processor(wav,sampling_rate=sr, return_tensors="pt")
        inputs=inputs.to(self.device)
        with torch.no_grad():
            outputs = self.speech_encoder(**inputs,output_hidden_states=True)
        states=outputs.hidden_states[self.layer].squeeze(0).cpu().numpy()
        return states
    
    def tokenize(self, wav, sr=16000):
        assert not self.pre_tokenized, "set pre_tokenized=False to load quantizer model"
        inputs=self.processor(wav,sampling_rate=sr, return_tensors="pt")
        inputs=inputs.to(self.device)
        with torch.no_grad():
            outputs = self.speech_encoder(**inputs,output_hidden_states=True)
        states=outputs.hidden_states[self.layer].squeeze(0).cpu().numpy()
        tokens = self.km_model.predict(states)
        return tokens
    
    @staticmethod
    def _collapse_tensor(tokens):
        is_continuous = tokens[1:]==tokens[:-1]
        return torch.cat([tokens[:1], tokens[1:][~is_continuous]])
    
    @staticmethod
    def _collapse_numpy(tokens):
        is_continuous = tokens[1:]==tokens[:-1]
        return np.concatenate([tokens[:1], tokens[1:][~is_continuous]])
    
    def _spm_decode(self, tokens):
        tokens=self.spm.DecodeIds(tokens)
        new_tokens=[]
        for t in tokens:
            t = ord(t)
            if t in self.reverse.keys():
                new_tokens.append(str(self.reverse[t]))
        decoded = ' '.join(new_tokens)
        return decoded
    
    def __call__(self, texts, **kwargs):
        token_idxs = [np.array(text.split(' ')).astype(int) for text in texts]
        if self.collapse:
            token_idxs = [self._collapse_numpy(tokens) for tokens in token_idxs]
        texts = [' '.join(tokens.astype(str)) for tokens in token_idxs]
        token_idxs = [list(tokens) for tokens in token_idxs]
        if self.spm is not None:
            token_idxs = [self.spm.Encode(''.join(chr(self.valid_unicode[t]) for t in tokens)) for tokens in token_idxs]
        token_idxs=[torch.tensor([self.blank] + tokens) for tokens in token_idxs]
        token_lengths = torch.tensor([len(idxs) for idxs in token_idxs])
        token_idxs = nn.utils.rnn.pad_sequence(token_idxs,batch_first=True, padding_value=self.pad_id)
        return token_idxs, token_lengths, texts
    
    def get_decoder(self):
        if self.spm is not None:
            return self._spm_decode #lambda tokens: ' '.join([str(t) for t in self._reverse(self.spm.DecodeIds(tokens))])
        else:
            return lambda tokens: ' '.join([str(t) for t in tokens])
    
    def get_remove_tokens(self):
        return [self.pad_id, self.blank]
    
    def pad_id(self):
        return self.pad_id
    
    def bos_id(self):
        return self.blank
    
    def eos_id(self):
        return self.blank
