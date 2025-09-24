import socket

HOST = "127.0.0.1"
PORT = 5000

# connettiti allo script esterno
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b"GET_RESULT")
    result = s.recv(1024).decode("utf-8")

# mostra il risultato a schermo
print("Risultato ricevuto:", result)

