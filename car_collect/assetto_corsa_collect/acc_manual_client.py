import socket
import pickle
import os
from car_foundation import CAR_FOUNDATION_MODEL_DIR
from termcolor import colored
import numpy as np
import datetime
import time


def start_controller(host='localhost', port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print(f"Connected to server at {host}:{port}")
        while True:
            data = s.recv(4096)
            [t, state]= pickle.loads(data)
            print(f"Received data: {[t, state]}")
            action = np.array([1., 1.], dtype=np.float32)
            # action *= 0.
            s.sendall(pickle.dumps(action))

if __name__ == "__main__":
    start_controller()
