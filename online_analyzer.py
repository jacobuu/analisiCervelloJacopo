import time
import numpy as np
from pylsl import StreamInlet, resolve_byprop
import joblib
import socket
import time

HOST = "127.0.0.1"  # localhost
PORT = 5000

# Carica il modello
model = joblib.load("csp_lda_model.pkl")

# Connetti agli stream
eeg_stream = resolve_byprop('type', 'EEG', timeout=2)
eye_stream = resolve_byprop('name', 'EyeTrackerStream', timeout=2)
marker_stream = resolve_byprop('name', 'RealShowImageMarkers', timeout=2)
if not eeg_stream or not eye_stream or not marker_stream:
    print("Non sono riuscito a trovare tutti gli stream LSL richiesti.")
    exit(1)
eeg_inlet = StreamInlet(eeg_stream[0], max_buflen=60)
eye_inlet = StreamInlet(eye_stream[0], max_buflen=60)
marker_inlet = StreamInlet(marker_stream[0])

# Parametri finestra
tmin, tmax = -0.2, 0.6  # secondi
sfreq = 256  # Hz (metti la frequenza reale EEG!)
win_samples = int((tmax - tmin) * sfreq)

# Buffers circolari
eeg_buffer = []  # lista di (timestamp, sample_vector)
eye_buffer = []

targets = {
    "banana": 0,
    "car": 0,
    "house": 0,
    "tree": 0,
    "face": 0,
    "cat": 0,
    "laptop": 0,
    "boat": 0,
}

print("In ascolto dei marker...")
session_over = False
while True:
    # Leggi EEG in continuo (aggiungi al buffer)
    samples, timestamps = eeg_inlet.pull_chunk(timeout=0.0)
    for s, ts in zip(samples, timestamps):
        eeg_buffer.append((ts, np.array(s)))
    # Mantieni solo ultimi 60s
    if len(eeg_buffer) > sfreq * 60:
        eeg_buffer = eeg_buffer[-sfreq*60:]

    # Leggi eye tracker (stessa logica se ti serve)
    samples, timestamps = eye_inlet.pull_chunk(timeout=0.0)
    for s, ts in zip(samples, timestamps):
        eye_buffer.append((ts, np.array(s)))
    if len(eye_buffer) > sfreq * 60:
        eye_buffer = eye_buffer[-sfreq*60:]

    # Controlla marker
    marker, ts = marker_inlet.pull_sample(timeout=0.0)
    if marker:
        if "ON" in marker[0]:
            if "/" in marker[0]:
                marker_label = marker[0].split("/")[1].split("_")[0]
            else:
                marker_label = marker[0].split("_")[0]
            print(f"Marker {marker_label} @ {ts:.3f}")

            # Prendi finestra EEG relativa al marker
            t_start, t_end = ts + tmin, ts + tmax
            epoch_data = [s for (t, s) in eeg_buffer if t_start <= t <= t_end]

            epoch_data = np.array(epoch_data).T  # shape (n_chans, n_times)
            if epoch_data.shape[1] != win_samples:
                print("⚠️ Epoch incompleto, salto.")
                continue

            # Adatta al modello
            X_epoch = epoch_data[np.newaxis, :, :]  # (1, n_chans, n_times)
            y_pred = model.predict(X_epoch)
            if y_pred[0] == 0:
                targets[marker_label] += 1
        if "END" in marker[0]:
            print("Sessione terminata.")
            session_over = True
    
    time.sleep(0.01)
    # Manda il risultato, il target con più voti
    

    if session_over:  # soglia di conferma
        prediction = max(targets, key=targets.get)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen(1)
            print(f"[Analyzer] In attesa di connessione su {HOST}:{PORT}...")
            conn, addr = s.accept()
            with conn:
                print("[Analyzer] Connesso a", addr)
                data = conn.recv(1024).decode("utf-8")
                if data == "GET_RESULT": 
                    conn.sendall(prediction.encode("utf-8"))
                    print("[Analyzer] Risultato inviato:", prediction)
                    session_over = False
                    targets = {k:0 for k in targets}  # resetta voti
                    print("In ascolto dei marker...")
                    time.sleep(0.5)  # evita di riaprire subito la socket
        # break  # esci dal loop se vuoi terminare il programma