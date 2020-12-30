# encoding: utf-8
"""
@author:  zhaominyi heyun
@contact: 20210240011@fudan.edu.cn 20210240010@fudan.edu.cn
"""
from client import Client
from config import Config
from PIL import ImageTk, Image
from tkinter import *
from time import sleep

# place the video
def place_video():
    fram = get_nxt_frame()
    if fram==None:
        return
    fram,_ = fram
    imgtk = ImageTk.PhotoImage(image=fram)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.update()


# get the frame
def get_nxt_frame():
    if not usr.is_receiving_rtp:
        return
    frame = usr.get_next_frame()
    if frame is not None:
        return frame

def connect_server():
    usr.establish_rtsp_connection()
    usr.send_setup_request()

root = Tk()
app = Frame(root, bg="white")
app.grid()
# Create a label in the frame
lmain = Label(app)
lmain.grid()

file_name = Config.DEFAULT_FILE
host_address = Config.HOST
host_port = Config.SERVER_PORT
rtp_port = host_port + 1
print("Start to run client")
usr = Client(file_name, host_address, host_port, rtp_port)
# try to connect the server
connect_server()

print("Start to play!")
# start to show the video
usr.send_play_request()
while True: 
    place_video()
    sleep(1/Config.DEFAULT_FPS)
root.mainloop()