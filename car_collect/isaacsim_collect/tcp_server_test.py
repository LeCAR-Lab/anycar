import socket
import pickle

def start_server():
    host='127.0.0.1'
    port=65432
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                received_list = pickle.loads(data)
                print(f"Received data: {received_list}")
                conn.sendall(pickle.dumps(received_list))

if __name__ == "__main__":
    start_server()