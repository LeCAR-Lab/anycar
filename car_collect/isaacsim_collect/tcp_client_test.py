import socket
import pickle

def start_client(host='127.0.0.1', port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print(f"Connected to server at {host}:{port}")
        while True:
            message = input("Enter a list of numbers separated by commas (or 'exit' to quit): ")
            if message.lower() == 'exit':
                break
            data_list = [int(num) for num in message.split(',')]
            s.sendall(pickle.dumps(data_list))
            data = s.recv(4096)
            received_list = pickle.loads(data)
            print(f"Received data: {received_list}")

if __name__ == "__main__":
    start_client()
