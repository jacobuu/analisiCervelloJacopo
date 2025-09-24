import socket
import time

HOST = "127.0.0.1"  # localhost
PORT = 5000

def run_analysis():
    """Simula analisi EEG e ritorna il target"""
    print("Analisi EEG in corso...")
    time.sleep(10)  # simulazione di calcoli lunghi
    return "stim_3"

if __name__ == "__main__":
    result = run_analysis()

    # avvia un server TCP
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[Analyzer] In attesa di connessione su {HOST}:{PORT}...")
        conn, addr = s.accept()
        with conn:
            print("[Analyzer] Connesso a", addr)
            data = conn.recv(1024).decode("utf-8")
            if data == "GET_RESULT":
                conn.sendall(result.encode("utf-8"))
                print("[Analyzer] Risultato inviato:", result)