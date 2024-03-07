import socket
from time import sleep
import signal		
import sys		

def signal_handler(sig, frame):
    print('Clean-up !')
    cleanup()
    sys.exit(0)

def cleanup():
    s.close()
    print("cleanup done")

ip = "192.168.10.139"     # Mobile
#ip = "192.168.1.35"        # Wifi

#To understand the working of the code, visit https://docs.python.org/3/library/socket.html
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ip, 8002))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            print(data.decode())  # Decode the received data
            #instructions = "FRFR"  
            instructions = "FFRLRRFRFLF"  
            conn.sendall(str.encode(instructions))
            sleep(1)
