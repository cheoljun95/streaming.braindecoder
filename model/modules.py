# check source code https://pytorch.org/audio/main/_modules/torchaudio/models/rnnt.html#RNNT
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchaudio.models.rnnt import _Predictor, _Joiner, _EmformerEncoder
from torchaudio.pipelines.rnnt_pipeline import ( _FeatureExtractor,
                                _FunctionalModule, _GlobalStatsNormalization,
                                _decibel, _gain)
import model.predictor as predictor
import model.joiner as joiner
import model.transcriber as transcriber
import model.feature_extractor as feature_extractor

import torchaudio

def build_predictor(MODEL=predictor.BasePredictor, **predictor_configs):
    if isinstance(MODEL, str):
        MODEL = getattr(predictor, MODEL)
    return MODEL(**predictor_configs)

def build_joiner(MODEL=joiner.BaseJoiner, **joiner_configs):
    if isinstance(MODEL, str):
        MODEL = getattr(joiner, MODEL)
    return MODEL(**joiner_configs)
    
def build_transcriber(MODEL=transcriber.BaseTranscriber, **transcriber_configs):
    if isinstance(MODEL, str):
        MODEL = getattr(transcriber, MODEL)
    return MODEL(**transcriber_configs)


def build_feature_extractor(MODEL=None, **feature_extractor_configs):
    if 'no_extractor' in feature_extractor_configs.keys() and feature_extractor_configs['no_extractor']:
        return None
    if MODEL is None:
        local_path = torchaudio.utils.download_asset("pipeline-assets/global_stats_rnnt_librispeech.json")
        return _ModuleFeatureExtractor(
                torch.nn.Sequential(
                    torchaudio.transforms.MelSpectrogram(
                        sample_rate=feature_extractor_configs['sample_rate'],
                        n_fft=feature_extractor_configs['n_fft'],
                        n_mels=feature_extractor_configs['n_mels'],
                        hop_length=feature_extractor_configs['hop_length']
                    ),
                    _FunctionalModule(lambda x: x.transpose(1, 0)),
                    _FunctionalModule(lambda x: _piecewise_linear_log(x * _gain)),
                    _GlobalStatsNormalization(local_path),
                    _FunctionalModule(lambda x: torch.nn.functional.pad(x, (0, 0, 0, feature_extractor_configs['_right_padding']))),
                )
            )
    else:
        if isinstance(MODEL, str):
            MODEL = getattr(transcriber, MODEL)
        return MODEL(**feature_extractor_configs)

def _piecewise_linear_log(x):
    e = torch.exp(torch.tensor(1)).to(x.device).to(x.dtype)
    x[x >e] = torch.log(x[x > e])
    x[x <= e] = x[x <= e] / e
    return x
    
class _ModuleFeatureExtractor(torch.nn.Module, _FeatureExtractor):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        
    def forward(self, input, input_lengths):
        """Generates features and length output from the given input tensor.

        Args:
            input (torch.Tensor): input tensor.

        Returns:
            (torch.Tensor, torch.Tensor):
            torch.Tensor:
                Features, with shape `(length, *)`.
            torch.Tensor:
                Length, with shape `(1,)`.
        """
        features = [self.pipeline(x[:l]) for x,l in zip(input, input_lengths)]
        lengths = torch.tensor([ft.shape[0] for ft in features])
        features = nn.utils.rnn.pad_sequence(features,batch_first=True, padding_value=0.0)
        return features, lengths