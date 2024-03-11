from cupyx.scipy.signal import hilbert, filtfilt
from scipy.signal import remez, resample
import cupy as cp
import numpy as np


class DenseHGA():
    def __init__(self, freq_range=[70,150], fs=1000, num_taps=351, ds_factor=5, average=True,):
        self.bands = [[b,b+1] for b in range(*freq_range)]
        self.filters = np.stack(self.get_filters(self.bands, fs=fs, num_taps=num_taps))
        self.filters = cp.asarray(self.filters)
        self.ds_factor = ds_factor
        self.average = average
        
    def get_filters(self, bands, fs,trans_width=5, num_taps=151):
        filters = []
        for band in bands:
            if band[1]+trans_width > 0.5*fs:
                edges = [0, band[0] - trans_width, band[0],0.5*fs]
                taps = remez(num_taps, edges, [0, 1], fs=fs)
            else:
                edges = [0, band[0] - trans_width, band[0], band[1],
                         band[1] + trans_width, 0.5*fs]
                taps = remez(num_taps, edges, [0, 1, 0], fs=fs)
            filters.append(taps)
        return filters
    def __call__(self, signals):
        # signals: (L, d)
        signals = cp.asarray(signals)
        filtered_signals = [] 
        for filter_ in self.filters:
            filtered_signals.append(filtfilt(filter_, [1],signals,axis=0))
        analytical_signals = []
        for filtered in filtered_signals:
            hga = cp.abs(hilbert(filtered, axis=0))
            hga_ds = hga[:int(len(hga)//self.ds_factor*self.ds_factor)].reshape(-1,self.ds_factor,hga.shape[-1]).mean(1)
            
            analytical_signals.append(hga_ds)
        analytical_signals = cp.stack(analytical_signals)
        if self.average:
            mean_hga = analytical_signals.mean(0).get()
        else:
            mean_hga = analytical_signals.get()
        
        return mean_hga
