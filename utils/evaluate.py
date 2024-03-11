import torch
import torch.nn as nn
import torchaudio
import numpy as np

def get_errors(pred, gt, average=True):
    
    cers = [torchaudio.functional.edit_distance(gt_.lower(), pred_.lower()) / len(gt_) for gt_, pred_ in zip(gt,pred)]
    wers = [torchaudio.functional.edit_distance(pred_.lower().split(' '), gt_.lower().split(' ')) / len(gt_.split(' ')) for gt_, pred_ in zip(gt,pred)]
    if average:
        outputs={'cer':np.mean(cers),
                 'wer':np.mean(wers),
                 #'nun':len(cers)
                }
    else:
        outputs={'cer':cers,
                 'wer':wers,
                }
    
    return outputs

def get_unit_errors(pred, gt, average=True):
    
    wers = [torchaudio.functional.edit_distance(pred_.lower().split(' '), gt_.lower().split(' ')) / len(gt_.split(' ')) for gt_, pred_ in zip(gt,pred)]
    if average:
        outputs={
                 'uer':np.mean(wers),
                 #'nun':len(cers)
                }
    else:
        outputs={'uer':wers,
                }
    
    return outputs
