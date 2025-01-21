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
And place them under `data` by simlink.

```
mkdir data
cd data
ln -s DATA_ROOT/LJSpeech-1.1/wavs ./
ln -s DATA_ROOT/LJSpeech-1.1/hb100 ./
cd ..
```

#### Train vocoder

Run the following command to train a vocoder. It will run the default config, `configs/lj_hificar_dur.yaml`. Please check the parameter there for customization. The above process should be done properly to run this code. The training checkpoints and logs will be created under `outputs/DATE/TIME`.

```
python train_vocoder.py
```
