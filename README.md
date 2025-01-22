# Streaming Brain2Speech Decoder by RNN-T

## Figures

* Reproduction of the main results of the manuscript are in `figures`

## Installation 

* Set up a new Conda environment
```
conda create -n rnnt python=3.9
conda activate rnnt
```

* Install pytorch (tested on torch=1.13.1, CUDA==11.7 and CUDA=12.2, Ubuntu 22.04.2 LTS, GPU support: NVIDIA A6000 with 49 GB memory)  This may take sometime (~10 mins) for downloading the source, which depends on the host's bandwidth. 
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

* Install dependency requirments. This will take less then 1~2 minutes. 
```
pip install -r requirements.txt
```

* The checkpoint of our main model should be in `model_ckpts/tm1k.ckpt`, and checkpoints for pretrained language models and vocoder should be under the `data/pretrained_modules` and `data/vocoder`, respectively.


## Download Dataset

The dataset is split to multiple `tar.gz` files. Please copy and paste `unzip_tar.sh` in this repo to the directory (lets say `DATA_ROOT`) where you downloaded all the zip files and run the script.
This will create `tm1k_mimed_slow` and 23628 numpy files will be extracted under the folder.
Then, come back to this source code directory, create data folder, and make symlink to the downloaded data 
```
cd data
ln -s DATA_ROOT/tm1k_mimed_slow ./tm1k_mimed_slow
cd ..
```


## Train model

For training 1024-word-General (tm1k) model, 

```
python train.py --config-name=tm1k
```
This will create output directory as `outputs/{DATE}/{TIME}/joint_hbbpe_tm1k`. The training log can be monitored by `tensorboard` (e.g., `cd outputs; tensorboard --logdir=./`).


After training is done, export and save trained model. It takes approximately 40 hours to reach the minimum error rates on a single NVIDIA A6000 GPU. CAUTION: this will overwrite the existing checkpoint file. Please use another name than `tm1k.ckpt` to avoid overwriting.
```
mkdir model_ckpts
python export_decoder.py --ckpt_path=outputs/{DATE}/{TIME}/joint_hbbpe_tm1k --output_path=model_ckpts/tm1k.ckpt
```
If `ckpt_path` is not specified, the most recent training directory under `outputs` will be processed.

## Run model on test data

For tm1k, you can run
```
python evaluate_model.py --config-name=tm1k
```
The results will be saved under `data/results/tm1k` by default. The parameters for the streaming system are specified in `eval_configs/tm1k.yaml`. There, the `rnnt_ckpt_path` and `test_files` and `save_path` are specified for specific model. This will save results under the specified `save_path`. Check `metrics` entry for the output types. This may take upto one hour to process with GPU support, otherwise it will take longer. 


For extracting saliance, run
```
python evaluate_salience.py --config-name=tm1k_salience

```

## Analysis figures

Please check notebooks under `figures`

## Demo 

Please check `notebooks/Streaming_Demo.ipynb`. This demo shows a simulation on real-time text and speech decoding, where speech is being decoded in real-time and stacked at the buffer. You can hear the synthesized speech after running the code. The text will be printed simultaneously while decoding. The model loading in the notebook would take several seconds (<30 sec) and the demo will run in less than 30 seconds on CPU. 

## Training vocoder

Check `synthesis/README.md` for the instruction for training a vocoder.

## Other technical tips

Our framework relies on the acoustic-speech units extracted by K Means on the HuBERT latent features. `extract_hubert_units.py` is used for extracting target units for brain decoder and input data for vocoder. Check arguments by `python extract_hubert_units.py --help`. (`--dedup` should be on for extracting targets for the brain decoder.)


