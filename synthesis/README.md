## Training Synthesizer

### Dataset preparation

#### Data download
Download [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) corpus. If unzipped, it will give the audio directory, `LJSpeech-1.1/wavs`.  

In the paper, we used [YourTTS](https://github.com/Edresson/YourTTS) to convert these audio samples to participant's voice. Other voice cloning software can be used instead. E.g., [Coqui](https://github.com/coqui-ai/TTS?tab=readme-ov-file#voice-conversion). Due to the privacy issue, we skip this process. 

#### HuBERT unit extraction

Run `extract_hubert_units.py` script in the parent directory to extract HuBERT units from audios.

```
python extract_hubert_units.py --data_path=DATA_ROOT/LJSpeech-1.1/wavs --save_path=DATA_ROOT/LJSpeech-1.1/hb100
```

#### 
