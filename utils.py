import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pyxdf
import mne
import torch

def step_resample_to_target(et_ts, et_vals, target_ts):
    et_ts = np.asarray(et_ts)
    et_vals = np.asarray(et_vals)
    target_ts = np.asarray(target_ts)
    idx = np.searchsorted(et_ts, target_ts, side='right') - 1
    idx[idx < 0] = 0
    return et_vals[idx]

def read_data(path):
    # Load XDF
    streams, _ = pyxdf.load_xdf(path, synchronize_clocks=True, dejitter_timestamps=True)
    types = [s['info']['name'][0] for s in streams]
    eyetracker = streams[types.index('EyeTrackerStream')]
    eeg        = streams[types.index('diadem_eeg')]
    markers    = streams[types.index('RealShowImageMarkers')]

    # Compute overlap window
    min_timestamp = max(eeg['time_stamps'][0], eyetracker['time_stamps'][0])
    max_timestamp = min(eeg['time_stamps'][-1], eyetracker['time_stamps'][-1])

    # Crop EEG + ET
    eeg_mask = (eeg['time_stamps'] >= min_timestamp) & (eeg['time_stamps'] <= max_timestamp)
    et_mask  = (eyetracker['time_stamps'] >= min_timestamp) & (eyetracker['time_stamps'] <= max_timestamp)

    eeg_ts = eeg['time_stamps'][eeg_mask]
    et_ts  = eyetracker['time_stamps'][et_mask]
    eeg_data_full = eeg['time_series'][eeg_mask]

    # Build EEG Raw
    eeg_srate = float(eeg['info']['nominal_srate'][0])
    eeg_ch_names = [ch['label'][0] for ch in eeg['info']['desc'][0]['channels'][0]['channel']]
    if eeg_data_full.shape[1] == len(eeg_ch_names) and 'TRIGGER' in eeg_ch_names[-1].upper():
        eeg_ch_names = eeg_ch_names[:-1]
        eeg_data_full = eeg_data_full[:, :-1]

    eeg_data = eeg_data_full.T / 1e6  # (n_ch, n_samp)
    info_eeg = mne.create_info(ch_names=eeg_ch_names, sfreq=eeg_srate, ch_types=['eeg']*len(eeg_ch_names))
    raw = mne.io.RawArray(eeg_data, info_eeg, verbose=False)

    # Eyetracker -> misc channel
    et_vals = eyetracker['time_series'][et_mask].astype(float).flatten()
    et_on_eeg = step_resample_to_target(et_ts, et_vals, eeg_ts) - 0.5
    et_on_eeg = (et_on_eeg * 500e-7) * -1
    et_info = mne.create_info(['eyetracker_open'], raw.info['sfreq'], ['misc'])
    et_raw = mne.io.RawArray(et_on_eeg[np.newaxis, :], et_info, verbose=False)
    raw.add_channels([et_raw], force_update_info=True)

    # EEG filtering
    raw.filter(1, 15., fir_design='firwin', picks='eeg', verbose=False)
    # raw.filter(1, 100., fir_design='firwin', picks='eeg', verbose=False)
    # raw.notch_filter(np.arange(50, 128, 50), fir_design='firwin', picks='eeg', verbose=False)

    # Annotations from markers
    marker_times = markers['time_stamps']
    marker_desc  = [str(m[0]) for m in markers['time_series']]
    annotations = mne.Annotations(onset=marker_times - eeg_ts[0],
                                  duration=[0] * len(marker_times),
                                  description=marker_desc)
    raw.set_annotations(annotations)
    return raw

def epoching(raw, tmin=0.0, tmax=0.8):
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    event_id_custom = {
        "T_ON": event_id["T_ON"],
        "NT_ON": event_id["NT_ON"],
    }

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id_custom,
        tmin=tmin,
        tmax=tmax,
        baseline=(None, 0),
        picks="eeg",
        preload=True,
        verbose=False
    )
    # epochs = epochs.crop(0.2, tmax)  # remove pre-stimulus baseline
    return epochs, event_id_custom

def add_noise(data, noise_level=0.01):
    noise = noise_level * torch.randn(*data.shape)
    return data + noise


def load_all_recordings(root_dir, pattern="sub-*/ses-*/eeg/*.xdf", exclude_ses = [], tmin=0.0, tmax=0.8, subject=None, drop_channels=None):
    """
    Walks root_dir/pattern, applies your read_data+epoching per file,
    returns stacked X, y and a list of file paths for traceability.
    """
    files = sorted(glob.glob(os.path.join(root_dir, pattern)))
    files = [f for f in files if all(ses not in f for ses in exclude_ses)]
    if not files:
        raise RuntimeError(f"No XDF files found under: {os.path.join(root_dir, pattern)}")

    X_list, y_list, used_files = [], [], []

    for p in files:
        if subject is not None and subject not in p:
            print(f"[skip] {p}: not session {subject}")
            continue
        try:
            raw = read_data(p)
            if drop_channels is not None:
                raw.drop_channels(drop_channels)

            epochs, event_id = epoching(raw, tmin=tmin, tmax=tmax)
            X = epochs.get_data()  # (trials, chans, times)
            y_events = epochs.events[:, -1]
            y = np.array([1 if e == event_id["NT_ON"] else 0 for e in y_events], dtype=np.int64)

            # # eliminate ET channel 
            # if first_n_chans is not None:
            #     if X.shape[1] < first_n_chans:
            #         print(f"[skip] {p}: only {X.shape[1]} channels (< {first_n_chans})")
            #         continue
            #     X = X[:, :first_n_chans, :]

            X_list.append(X.astype(np.float32))
            y_list.append(y)
            used_files.append(p)
            print(f"[ok] {p}: epochs={len(y)}, chans={X.shape[1]}")

        except Exception as e:
            print(f"[error] {p}: {e}")

    if not X_list:
        raise RuntimeError("No valid recordings after parsing.")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y, used_files