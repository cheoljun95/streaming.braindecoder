# RNN-T Implementation for Arbtrary Units

## Installation 

* Set up a new Conda environment
```
conda create -n rnnt python=3.9
conda activate rnnt
```

* Install pytorch (tested on torch=1.13.1, CUDA==11.7, Linux)
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

* Install dependency requirments.
```
pip install -r requirements.txt
```

* Download SD-HuBERT checkpoints and quantizers from [here](https://drive.google.com/file/d/19v8E8hFDap-CwDdDD64fKuPfL66n1Hav/view?usp=sharing) and check `sdhubert_asset` folder.

* For the case using Peter's KMeans models which are trained with TorchPQ. Change the cuda version of cupy according to your own cuda version.

```
pip install cupy-cuda117
pip install torchpq
```

## Setting up syllable segmentation by SD-HuBERT

We use the segmentation algorithm suggested by [Peng et al., 2023](https://arxiv.org/abs/2305.11435). We are using the implementation shared by the author, so please check the original code/installation [here](https://github.com/jasonppy/syllable-discovery/tree/master). You need Cython for this.
```
cd ./mincut
python setup.py build_ext --inplace
```


## Supported model

* The transcriber models are using CNN+LSTM models with bidirectional connections. Check model/tokenizer.py for the label. The index for \<PAD> is *vacab size* and \<BLANK> is *vocab size* +1. This \<BLANK> is the blank used in RNNTLoss. For example, for hubert 100 units, there are total 102 labels that is 100 + \<PAD> + \<BLANK>. 

* `nocollapse` means that sequential repetitions in target lables are not collapsed. When collapsed, for example, a pseudo-text 10 11 11 11 21 32 32 32 21 becomes 10 11 21 32 21. If not annotated, the collapsing strategy is used.

* Please check `configs` and `model/tokenizer.py` for the configuration detail.


| Target Unit                | Test-UER | Test-WER | 
|----------------------------|:--------:|:--------:|
| grapheme                   |    10    |    21    |
| phoneme                    |    13    |    --    |
| phoneme-wstress            |    12    |    --    |
| phoneme-wspace             |    12    |    --    |
| hubert-l6_km100            |    21    |          |
| hubert-l6_km100_nocollapse |    19    |          |
| hubert-large-l6_pwkm200_nocollapse |    22    |          |



## Load pretrained model

* The checkpoints for RNNT modules, language model (predictor) and joiner, can be find in [here](https://drive.google.com/drive/folders/1GW1sHHzZEUv_vOhoaweE9daB65Ryk4P7). The models are trained using LibriSpeech training set (960h). Please download whatever you need and load the model by the below codes. These modules are basically training the same configuration of Emformer by PyTorch official implementation from `torchaudio.models.rnnt`. Please check `model/predictor.py` and `model/joiner.py` for more information.

* You can import and load modules by the following codes.

```python
from model.modules import build_predictor, build_joiner, build_transcriber, build_feature_extractor

module_ckpt_path = 'ckpts/grapheme_predictor.ckpt' # example path for grapheme LM.
module_ckpt = torch.load(module_ckpt_path)
predictor = build_predictor(**module_ckpt['config'])
predictor.load_state_dict(module_ckpt['state_dict'])
```

## Create LibriSpeech Transcriptions
* It is already created in `vox` and `lingua`, so you can skip it if you are using those servers.
For `vox`, LibriSpeech dataset is in `/data/cheoljun/librispeech/LibriSpeech`. For `lingua`, it is in `/data/common/LibriSpeech`.
```
python scripts/create_text_transctipt.py --data_root={LibriSpeech path}
```

## Extract HuBERT units from LibriSpeech 

* It will automatically download HuBERT and Kmeans model to quantize speech. The labels will be created under `LibriSpeech/unit_label/hubert-l6_km{km_n}`, with the same convention as LibriSpeech audio files.
```
cd scripts
python extract_hubert_units.py --data_root={LibriSpeech path} --km_n={50 or 100 or 200}
```
Then, we can create transcriptions from the extracted units.
```
python create_hubert_transctipt.py --data_root={LibriSpeech path} -unit_tag=hubert-l6_km{km_n}
```


## Training
* You may want to change configuration in `yaml` file (e.g., the root directory for the data). Also, it can be specified through the command line, for example, `data.root_dir={LibriSpeech path}`
```
python train.py --config-name=cnnlstm_rnnt.yaml
```

* The code base is based on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). So check documentation if you want to add any new features in training (e.g., early stopping, learning rate scheduling, etc). Current logging is using Tensorboard and you can set Wandb instead by https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#wandb. 

## Evaluation

```
python evaluate.py  --output_path={training_output_paths}
```

## Export pretrained module

* The following command will save pretrained weights and config for each module of Predictor, Joiner and Transcriber. The checkpoint file will be saved under `ckpts` folder.
```
python export.py  --ckpt_path={training_output_paths}
```



## Train SentencePiece on HuBERT units

* Create one unicode dump file from HuBERT unit transcriptions.
```
python scripts/create_hubert_unicodes.py --save_path=/data/common/LibriSpeech/hubert-l6_km100_transcription
```

* Train SentencePiece with `spm_train` command. You can install this with `sudo apt install sentencepiece`. Note that the unicode normalization should be turned off for HuBERT units by setting `--normalization_rule_name=identity`.
```
mkdir spm
cd spm
spm_train --normalization_rule_name=identity --input=/data/common/LibriSpeech/hubert-l6_km100_transcription/train-all_unicodes.txt --model_prefix=hubert-l6_km100_bpe4096 --vocab_size=4096  --model_type=bpe
``` 

* Or you can download pretrained SentencePiece models on HuBERT units from [here](https://drive.google.com/file/d/1MwdHSf8Tnc7wl4jtcnqtD-0BPKqMJvny/view?usp=drive_link). Unzip it and check `spm` folder.

* Unit lengths of different combination of HuBERT KMeans cluster number and BPE/unigram(UNI) vocab size (in milisecond).

| Quantizer  | None      | BPE1024  | BPE2048  | BPE4096   | UNI1024 | UNI2048 | UNI4096 |
|-----|---------|----------|----------|-----------|---------|---------|---------|
| hubert-l6_km50  | 45&plusmn;34 | 117&plusmn;84 | 131&plusmn;92 | 146&plusmn;100 | 120&plusmn;87 | 136&plusmn;96 | 152&plusmn;105 |
| hubert-l6_km100 | 39&plusmn;29 |  64&plusmn;50 |  68&plusmn;54 | 70&plusmn;57  | 64&plusmn;51 | 67&plusmn;55 | 69&plusmn;58 |
| hubert-l6_km200 | 36&plusmn;25 |  55&plusmn;42 | 59&plusmn;46 | 62&plusmn;50   | 55&plusmn;43 | 59&plusmn;48 | 62&plusmn;52 |

<!---## TODO--->

<!---* ~~Train SentencePiece on HuBERT units. Maybe https://discuss.huggingface.co/t/training-sentencepiece-from-scratch/3477 or https://github.com/google/sentencepiece/blob/master/doc/options.md are useful.~~--->
