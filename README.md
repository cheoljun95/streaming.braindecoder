# Streaming Brain2Speech Decoder by RNN-T

## Figures

* Reproduction of the main results of the manuscript are in `figures`

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


## Download Dataset
Download the data package to `{DATA_ROOT}` and make a symlink to this directory. Every code is based on the relative path from this directory, so this symlink is required.
```
ln -s {DATA_ROOT} ./data
```

## Train model

For training 1024-word-General (tm1k) model, 

```
python train.py --config-name=tm1k
```
This will create output directory as `outputs/{DATE}/{TIME}/joint_hbbpe_tm1k`. The training log can be monitored by `tensorboard` (e.g., `cd outputs; tensorboard --logdir=./`).


After training is done, export and save trained model.
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
The results will be saved under `data/results/tm1k` by default. The parameters for the streaming system are specified in `eval_configs/tm1k.yaml`. There, the `rnnt_ckpt_path` and `test_files` and `save_path` are specified for specific model. This will save results under the specified `save_path`. Check `metrics` entry for the output types. 


For extracting saliance, run
```
python evaluate_salience.py --config-name=tm1k_salience

```
