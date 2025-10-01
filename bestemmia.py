import numpy as np
import matplotlib.pyplot as plt
import mne 

all_channels = ['AF7', 'Fp1', 'Fp2', 'AF8', 'F3', 'F4', 'P3', 'P4', 'P7', 'O1', 'O2', 'P8']
data = np.load('epoch_raw.npy', allow_pickle=True)[0,:,:] / 1e6  # convert to Volts

info = mne.create_info(ch_names=all_channels, sfreq=256, ch_types=['eeg']*len(all_channels))
epochs = mne.io.RawArray(data, info, verbose=False)

# lowpass
epochs.filter(None, 40., fir_design='firwin', picks='eeg', verbose=False)

epochs.plot(n_channels=len(all_channels), scalings='auto', show=True, block=True)