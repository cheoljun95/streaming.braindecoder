## BRAVO Model Commands

### Training

For training models, run
```
python train_bravo_multirnnt.py --config-name={CONFIG}
```

There are precomposed configs in `configs/`. These are for joint training decoder on HuBERT 100 units and 4096 BPE of charactors,using the data from Fall 2022.
```
# For unidirectional, causal model
joint_hbbpe_50ph.yaml
joint_hbbpe_500ph.yaml
joint_hbbpe_tm1k_synth.yaml

# For bidirectional, full-context model
fc_joint_hbbpe_50ph.yaml
fc_joint_hbbpe_500ph.yaml
fc_joint_hbbpe_tm1k_synth.yaml

# For ablating regions - ['superiortemporal', 'supramarginal', 'middletemporal']
ablate_joint_hbbpe_tm1k_synth.yaml
```



For the recent data, there are two configs for joint training with BPE or phoneme.
```
joint_hbbpe_tm1k_20231206.yaml # with BPE
joint_hbph_tm1k_20231206.yaml # with phoneme
ablate_joint_hbbpe_tm1k_20231206.yaml # For ablating regions - ['superiortemporal', 'supramarginal', 'middletemporal']
```
For example, if you want to train tm1k data from 2022 Fall, 

```
python train_bravo_multirnnt.py --config-name=joint_hbbpe_tm1k_synth
```

This will automatically create a directory under `outputs/`, wich will look like

```
outputs/{YYYY-MM-DD}/{HH-MM-SS}/{EXPERIMENT_NAME}
```

The `EXPERIMENT_NAME` is the one appears as `experiment_name` entity in the configuration file, which is mostly the same as config name.

### Training for new upcoming data

You can copy `joint_hbbpe_tm1k_20231206.yaml` and change the fields accordingly, or you can run

```
python train_bravo_multirnnt.py --config-name=joint_hbbpe_tm1k_20231206.yaml experiment_name=joint_hbbpe_tm1k_{NEW_DATE} data.train_files={TRAIN_FILES} data.test_files={TEST_FILES}
```


### Other training parameter

#### GPU

Modify `gpus` value in the config or put parameter as 
```
python train_bravo_multirnnt.py --config-name={CONFIG} gpus={GPU NUM: ex) 3}
```

#### Resuming training

Put output directory into the `resume_ckpt` field or
```
python train_bravo_multirnnt.py --config-name={CONFIG} --resume_ckpt={working directory}/outputs/{YYYY-MM-DD}/{HH-MM-SS}/{EXPERIMENT_NAME} data.test_files
```

#### Checkpointing

The metrics to track the best checkpoints are specified in the `log_metrics` field, 
For example in the HB100 and BPE joint training, I put as follows in the config file.
```
log_metrics: 
  - 'bpe_rnnt_loss'
  - 'unit_rnnt_loss'
  - 'bpe_uer'
  - 'unit_uer'
```

So, the training will keep track on the best checkpoints with those metric terms. (For phoneme, it will be `phoneme_` instead of `bpe_`).

Also, for the early stopping, `earlystop_metric` denotes which metric to determine early stopping and `patience` denotes the number of iterations upto which you decide keep training without improvement. 

#### Training/test data

The training and test files are specified in `data` field as `train_files` and `test_files` subfields. The `use_test_for_valid` parameter determines whether to use test trials as validation trials for training, which will report the metrics on. This field is set as `False` by default,and in that case, the last 5% of the training trials will be used as validation trials. For F22 data, this field was turned off so the 5% of the training data were used for the validation. 


### Exporting the package

To export the package run the following command.
```
python export_modules_generic.py --ckpt_path=outputs/{YYYY-MM-DD}/{HH-MM-SS}/{EXPERIMENT_NAME} --mode={MODE} --term={METRIC}
```

The `--ckpt_path` should be pointed to the output dir that is created by the training, and the `--mode` has two choices: `latest` or `best`. The former loads the latest checkpoints and the best loads the best checkpoitns along the training. For the best mode, you can specify the metric term to use.
A full example is as bellow.
```
python export_modules_generic.py --ckpt_path=outputs/2023-11-14/20-50-59/joint_tm1k_synthesis --mode=best --term=bpe_uer_loss
```

This will create ckpt file under `ckpts_generic` folder with the name as `{EXPERIMENT_NAME}.ckpt`. The resulting checkpoint file can be used for the `rnnt_ckpt` in the streaming package.

Please check the `notebooks/Streaming_Demo.ipynb` to run the streaming package.

### Evaluating

Once the training is done, you can run the evaluation script. The evaluation script is based on running the real-time streamer pipeline, which is specified in `eval_configs/default.yaml`.

```
python evaluate_bravo_rnnt.py --config-name=default training_output_path={working directory}/outputs/{YYYY-MM-DD}/{HH-MM-SS}/{EXPERIMENT_NAME} result_path={RESULT_SAVE_PATH}
```

You need to designate the checkpoint directory created by training script and the result path to save the evaluation results. You can specify the loading configuration as the same as exporting secion. You can put `latest` or `best` in `load_mode` field, and specify the metric by `loss_term`. The default is `best` and `bpe_uer`. 

But if you trained a phoneme version which doesn't have `bpe_uer`, you can put argument as `phoneme_uer` instead.

```
python evaluate_bravo_rnnt.py --config-name=default training_output_path={working directory}/outputs/{YYYY-MM-DD}/{HH-MM-SS}/{EXPERIMENT_NAME} result_path={RESULT_SAVE_PATH} load_mode=best loss_term=phoneme_uer 
```

Please note that the `default.yaml` is based on the real-time streaming wiht causal RNN-T model. So if you want to test a bidirectional model, you need to put an extended buffer size as an argument.

```
python evaluate_bravo_rnnt.py --config-name=default training_output_path={working directory}/outputs/{YYYY-MM-DD}/{HH-MM-SS}/{EXPERIMENT_NAME} result_path={RESULT_SAVE_PATH} buffer_size=1800
```

The below is an example of actual command for evaluating a bidirectional model trained on HB100 and BPE jointly using TM1K data from Fall 2022.

```
python evaluate_bravo_rnnt.py --config-name=default training_output_path=/home/cheoljun/rnnt.streaming.anyunit/outputs/2023-12-08/06-43-27/fc_joint_hbbpe_tm1k_synth    result_path=/home/cheoljun/rnnt.streaming.anyunit/bravo_eval_results buffer_size=1800
```

In the result paths, you will find a plot with WER using BPE, Synthesized+ASR, Non-expanded synsthesized+ASR, Synthesized ground-truth+ASR. It also saves the numbers for each trials as `wers.npy` and `predictions.npy`. Especially in `predictions.npy`, it also reports the first emission time.

