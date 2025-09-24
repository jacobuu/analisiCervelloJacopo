import numpy as np
import matplotlib.pyplot as plt
import pyxdf
import mne
from sklearn.model_selection import train_test_split
import torch
from braindecode.models import EEGNetv4

from braindecode import EEGClassifier
from skorch.helper import SliceDataset
from sklearn.metrics import accuracy_score



def step_resample_to_target(et_ts, et_vals, target_ts):
    """
    Map a step-like (0/1) signal to arbitrary target timestamps by
    taking the last known ET value at or before each target time
    (zero-order hold). If no prior sample exists, use the first.
    """
    et_ts = np.asarray(et_ts)
    et_vals = np.asarray(et_vals)
    target_ts = np.asarray(target_ts)

    idx = np.searchsorted(et_ts, target_ts, side='right') - 1
    idx[idx < 0] = 0
    return et_vals[idx]

def read_data(path):
    # Load XDF
    streams, _ = pyxdf.load_xdf(path, synchronize_clocks=True, dejitter_timestamps=True)
    print(f"Streams keys: {[s['info']['name'][0] for s in streams]}")
    def pick(name):
        for s in streams:
            if s['info']['name'][0] == name:
                return s
        raise RuntimeError(f"Stream '{name}' not found")

    # stream indices (controlla l'ordine nei tuoi file!)
    eyetracker = streams[1]
    eeg        = streams[3]
    markers    = streams[2]  # qui ci sono i tuoi marker

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
    raw = mne.io.RawArray(eeg_data, info_eeg)

    # Eyetracker -> misc channel
    et_vals = eyetracker['time_series'][et_mask].astype(float).flatten() 
    et_on_eeg = step_resample_to_target(et_ts, et_vals, eeg_ts) - 0.5
    et_on_eeg = (et_on_eeg * 500e-7) * -1
    et_info = mne.create_info(['eyetracker_open'], raw.info['sfreq'], ['misc'])
    et_raw = mne.io.RawArray(et_on_eeg[np.newaxis, :], et_info)
    raw.add_channels([et_raw], force_update_info=True)

    # EEG filtering
    raw.filter(1., 100., fir_design='firwin', picks='eeg')
    raw.notch_filter(np.arange(50, 128, 50), fir_design='firwin', picks='eeg')

    # ----------------------------
    # ADD MARKERS AS ANNOTATIONS
    # ----------------------------
    marker_times = markers['time_stamps']
    marker_desc  = [str(m[0]) for m in markers['time_series']]  # stringhe
    # durata zero = istantaneo
    annotations = mne.Annotations(onset=marker_times - eeg_ts[0],
                                  duration=[0] * len(marker_times),
                                  description=marker_desc)
    raw.set_annotations(annotations)

    return raw

def epoching(raw):
    # Converti annotations → eventi
    events, event_id = mne.events_from_annotations(raw)

    print("Event IDs:", event_id)  
    # Controlla che T_ON e NT_ON compaiano qui

    # Definisci solo quelli che ti servono
    event_id_custom = {
        "T_ON": event_id["T_ON"],
        "NT_ON": event_id["NT_ON"],
    }

    # Epoching: da onset evento (t=0) a 0.8 secondi dopo
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id_custom,
        tmin=0.0,
        tmax=0.5,
        baseline=None,  # oppure (0,0.1) se vuoi baseline correction
        picks="eeg",
        preload=True,
    )

    # print(epochs)
    # epochs.plot(n_epochs=10, scalings=dict(eeg=100e-6))
    return epochs, event_id_custom

def add_noise(data, noise_level=0.01):
    # use torch
    noise = noise_level * torch.randn(*data.shape)
    return data + noise

if __name__ == "__main__":
    path = "eegAcquisitions\\sub-P001\\ses-S001\\eeg\\sub-P001_ses-S001_task-Default_run-002_eeg.xdf"
    raw = read_data(path)
    epochs, event_id = epoching(raw)

    # Dati dagli epochs
    X = epochs.get_data()  # shape: (n_trials, n_channels, n_times)
    X = X[:, :12, :]
    y = epochs.events[:, -1]

    # Mappa T_ON = 1, NT_ON = 0
    y = np.array([1 if e == event_id["T_ON"] else 0 for e in y])
    print("X shape:", X.shape, "y shape:", y.shape)

    # balance classes
    class0_indices = np.where(y == 0)[0]
    class1_indices = np.where(y == 1)[0]
    min_class_size = min(len(class0_indices), len(class1_indices))
    balanced_indices = np.concatenate([
        np.random.choice(class0_indices, min_class_size, replace=False),
        np.random.choice(class1_indices, min_class_size, replace=False)
    ])
    X = X[balanced_indices]
    y = y[balanced_indices]


    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Converti in torch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test  = torch.tensor(y_test, dtype=torch.long)

    


    # normalize the X data
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # upsample train and test
    repeat_factor = 5
    # add noise to each repetition
    X_train = torch.cat([add_noise(X_train, noise_level=0.1) for _ in range(repeat_factor)], axis=0)
    y_train = y_train.repeat(repeat_factor)

    # X_test = torch.cat([add_noise(X_test, noise_level=0.1) for _ in range(repeat_factor)], axis=0)
    # y_test = y_test.repeat(repeat_factor)
    # print("After upsampling, X_train shape:", X_train.shape, "y_train shape:", y_train.shape)


    # Braindecode vuole input (batch, channels, time)
    print("y=0 in train:", (y_train==0).sum().item(), "y=1 in train:", (y_train==1).sum().item())
    print("y=0 in test:", (y_test==0).sum().item(), "y=1 in test:", (y_test==1).sum().item())

    n_channels = X.shape[1]
    n_times = X.shape[2]
    n_classes = 2

    model = EEGNetv4(
        n_chans=n_channels,
        n_outputs=n_classes,
        input_window_samples=n_times,
        final_conv_length="auto",  # calcolato automaticamente
    )
    print(model)


    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=1e-3,
        max_epochs=200,
        batch_size=32,
        iterator_train__shuffle=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Braindecode usa skorch, quindi i dati vanno wrappati in dataset
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    print("Train accuracy:", acc)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy:", acc)

    # raw.plot(scalings=dict(eeg_data=100e-6, eyetracker_open=100e-6), n_channels=32, duration=10)
    # raw.plot(scalings=dict(eeg=100e-6, misc=100e-6), n_channels=32, duration=10)
    # input("Press Enter to close the plot...")
    # plt.close()