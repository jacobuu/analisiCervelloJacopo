import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pyxdf
import mne
from sklearn.model_selection import train_test_split
import torch
from braindecode.models import EEGNetv4
from braindecode import EEGClassifier
from sklearn.metrics import accuracy_score
from skorch.callbacks import EarlyStopping


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
    raw.filter(1., 100., fir_design='firwin', picks='eeg', verbose=False)
    raw.notch_filter(np.arange(50, 128, 50), fir_design='firwin', picks='eeg', verbose=False)

    # Annotations from markers
    marker_times = markers['time_stamps']
    marker_desc  = [str(m[0]) for m in markers['time_series']]
    annotations = mne.Annotations(onset=marker_times - eeg_ts[0],
                                  duration=[0] * len(marker_times),
                                  description=marker_desc)
    raw.set_annotations(annotations)
    return raw

def epoching(raw):
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    event_id_custom = {
        "T_ON": event_id["T_ON"],
        "NT_ON": event_id["NT_ON"],
    }

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id_custom,
        tmin=0.0,
        tmax=0.8,
        baseline=None,
        picks="eeg",
        preload=True,
        verbose=False
    )
    return epochs, event_id_custom

def add_noise(data, noise_level=0.01):
    noise = noise_level * torch.randn(*data.shape)
    return data + noise


def load_all_recordings(root_dir, pattern="sub-*/ses-*/eeg/*.xdf", first_n_chans=12):
    """
    Walks root_dir/pattern, applies your read_data+epoching per file,
    returns stacked X, y and a list of file paths for traceability.
    """
    files = sorted(glob.glob(os.path.join(root_dir, pattern)))
    if not files:
        raise RuntimeError(f"No XDF files found under: {os.path.join(root_dir, pattern)}")

    X_list, y_list, used_files = [], [], []

    for p in files:
        if "P001" not in p:
            print(f"[skip] {p}: not session 001")
            continue
        try:
            raw = read_data(p)
            epochs, event_id = epoching(raw)
            X = epochs.get_data()  # (trials, chans, times)
            y_events = epochs.events[:, -1]
            y = np.array([1 if e == event_id["T_ON"] else 0 for e in y_events], dtype=np.int64)

            # eliminate ET channel 
            if first_n_chans is not None:
                if X.shape[1] < first_n_chans:
                    print(f"[skip] {p}: only {X.shape[1]} channels (< {first_n_chans})")
                    continue
                X = X[:, :first_n_chans, :]

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

if __name__ == "__main__":
    # -------- set your root once; wrapper will read them all --------
    root = "eegAcquisitions"  # e.g., contains sub-P001/ses-S001/... etc.
    X, y, files = load_all_recordings(root, pattern="sub-*/ses-*/eeg/*.xdf", first_n_chans=12)
    print("STACKED:", X.shape, y.shape, f"from {len(files)} files")

    # balance classes (same as your single-file flow)
    class0_indices = np.where(y == 0)[0]
    class1_indices = np.where(y == 1)[0]
    min_class_size = min(len(class0_indices), len(class1_indices))
    balanced_indices = np.concatenate([
        np.random.choice(class0_indices, min_class_size, replace=False),
        np.random.choice(class1_indices, min_class_size, replace=False)
    ])
    X = X[balanced_indices]
    y = y[balanced_indices]

    # Split train/test (keep your original split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # To torch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test  = torch.tensor(y_test, dtype=torch.long)

    # normalize (your logic)
    # mean = X_train.mean(dim=0, keepdim=True)
    # std = X_train.std(dim=0, keepdim=True)
    # std[std == 0] = 1.0
    # X_train = (X_train - mean) / std
    # X_test = (X_test - mean) / std

    # X_train = (X_train - X_train.mean(dim=2, keepdim=True)) / X_train.std(dim=2, keepdim=True)
    # X_test = (X_test - X_test.mean(dim=2, keepdim=True)) / X_test.std(dim=2, keepdim=True)
    # upsample train with noise (same as you)
    # repeat_factor = 5
    # X_train = torch.cat([add_noise(X_train, noise_level=0.1) for _ in range(repeat_factor)], axis=0)
    # y_train = y_train.repeat(repeat_factor)

    print("y=0 in train:", (y_train==0).sum().item(), "y=1 in train:", (y_train==1).sum().item())
    print("y=0 in test:", (y_test==0).sum().item(), "y=1 in test:", (y_test==1).sum().item())

    n_channels = X.shape[1]
    n_times = X.shape[2]
    n_classes = 2

    model = EEGNetv4(
        n_chans=n_channels,
        n_outputs=n_classes,
        input_window_samples=n_times,
        final_conv_length="auto",
        drop_prob=0.5
    )
    print(model)

    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=1e-4,
        max_epochs=500,
        batch_size=64,
        iterator_train__shuffle=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=[EarlyStopping(patience=50)]
    )

    clf.fit(X_train, y_train)
    # plot the loss
    plt.plot(clf.history[:, 'train_loss'])
    plt.plot(clf.history[:, 'valid_loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Valid"])
    plt.show()

    # same quick metrics
    y_pred = clf.predict(X_train)
    print("Train accuracy:", accuracy_score(y_train.numpy(), y_pred))
    y_pred = clf.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test.numpy(), y_pred))
