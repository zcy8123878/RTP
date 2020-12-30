# encoding: utf-8

import io
import socket
import tkinter

from PIL import ImageGrab
from PIL import Image

from lib.pymovie_stream import VideoStream
from lib.rtsp_packet import RTSPPacket
from lib.rtp_packet import RTPPacket
from config import Config
from threading import Thread
from time import sleep
import time

from utils import Executor
from pynput import keyboard

class Server:
    # the time server send frame
    DPS = 1000//Config.DEFAULT_FPS

    SESSION_ID = Config.SESSION_ID

    DEFAULT_HOST = Config.HOST
    DEFAULT_PORT = Config.SERVER_PORT
    DEFAULT_CHUNK_SIZE = Config.CHUNK_SIZE

    RTSP_SOFT_TIMEOUT = Config.TIMEOUT
    
    
    class STATE:
        INIT = 0
        PAUSED = 1
        PLAYING = 2
        FINISHED = 3
        TEARDOWN = 4
        
    class MODEL:
        DEFAULT_MODEL = 0
        MODEL1 = 1
        MODEL2 = 2
        MODEL3 = 3
    
    def __init__(self):
        self.server_state = self.STATE.INIT
        self.executor = Executor()
        model_dict = {1:'pretrained/starry_night.onnx',
                      2:'pretrained/transform_net_31_73.onnx',
                      3:'pretrained/transform_net_31_98.onnx'}
        self.executor.load_model(model_dict, mtype='onnx')
        self.model = self.MODEL.DEFAULT_MODEL
        return
        
    def monitor_keyboard(self):
        """
        监听键盘的线程 完成模型的选择
        :return:
        """
        self._monitor_keyboard_thread = Thread(target=self.keyboard_response)
        self._monitor_keyboard_thread.setDaemon(True)
        self._monitor_keyboard_thread.start()
        return

    def keyboard_response(self):

        def on_press(key):
            # 按下按键时执行。
            try:
                print('alphanumeric key {0} pressed'.format(
                    key.char))

                if key.char == '1':
                    self.model = self.MODEL.MODEL1
                elif key.char == '2':
                    self.model = self.MODEL.MODEL2
                elif key.char == '3':
                    self.model = self.MODEL.MODEL3
                else:
                    self.model = self.MODEL.DEFAULT_MODEL

            except AttributeError:
                print('special key {0} pressed'.format(
                    key))
            # 通过属性判断按键类型。

        # Collect events until released
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

        return

    def setup(self):
        self.wait_for_connection()
        self.do_setup()
        return


    # wait for the clint
    def wait_for_connection(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        address = self.DEFAULT_HOST, self.DEFAULT_PORT
        s.bind(address)
        print(f"Start Server at {address[0]}:{address[1]} , waiting for client")
        s.listen(1)
        self.rtsp_connection, self.client_address = s.accept()
        self.rtsp_connection.settimeout(self.RTSP_SOFT_TIMEOUT/1000.)
        print(f"Accept {self.client_address[0]}:{self.client_address[1]}")
        return
    
    # wait for the user to start
    def do_setup(self):
        if self.server_state != self.STATE.INIT:
            raise Exception('server is already setup')
        while True:
            packet = self.get_rtsp_packet()
            if packet.request_type == RTSPPacket.SETUP:
                self.server_state = self.STATE.PAUSED
                print('State set to PAUSED')
                self.client_address = self.client_address[0], packet.rtp_dst_port
                self.setup_rtp(packet.video_file_path)
                self.send_rtsp_response(packet.sequence_number)
                break
        return
    
    # wait for rtsp packet
    def get_rtsp_packet(self):
        return RTSPPacket.from_request(self.rtsp_recv())
    
    def rtsp_recv(self, size=DEFAULT_CHUNK_SIZE) -> bytes:
        recv = None
        while True:
            try:
                recv = self.rtsp_connection.recv(size)
                break
            except socket.timeout:
                continue
        print(f"Received from client: {repr(recv)}")
        return recv
    
    def setup_rtp(self,url):
        print(f"Get video url: {url}")
        self.video_stream = VideoStream(url)
        self.rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.start_rtp_send_thread()
        return
    
    def start_rtp_send_thread(self):
        self._rtp_send_thread = Thread(target=self.do_screen_send)
        self._rtp_send_thread.setDaemon(True)
        self._rtp_send_thread.start()
        return
    
    def do_video_send(self):
        print(f"Start sending video to {self.client_address[0]}:{self.client_address[1]}")
        while True:
            if self.server_state == self.STATE.TEARDOWN:
                return
            if self.server_state != self.STATE.PLAYING:
                sleep(0.5)  # diminish cpu hogging
                continue
            if self.video_stream.current_frame_number >= self.video_stream.max_frame_number:  # frames are 0-indexed
                print('Reached end of file.')
                self.server_state = self.STATE.FINISHED
                return
            frame = self.video_stream.get_next_frame()
            frame_number = self.video_stream.current_frame_number
            rtp_packet = RTPPacket(
                payload_type=RTPPacket.TYPE.MJPEG,
                sequence_number=frame_number,
                timestamp=frame_number*self.DPS,
                payload=frame
            )
            print(f"Sending packet #{frame_number}")
            print('Packet header:')
            rtp_packet.print_header()
            packet = rtp_packet.get_packet()
            self.send_rtp_packet(packet)
            sleep(self.DPS/1000.)
        return

    def do_screen_send(self):
        print(f"Start sending screen to {self.client_address[0]}:{self.client_address[1]}")
        DEFAULT_SIZE = (320 * 2, 200 * 2)

        # win = tkinter.Tk()
        # width = win.winfo_screenwidth() # can not get full screen size on my PC
        # height = win.winfo_screenheight()

        frame_number = -1

        while True:
            if self.server_state == self.STATE.TEARDOWN:
                return
            if self.server_state != self.STATE.PLAYING:
                sleep(0.5)  # diminish cpu hogging
                continue

            t1 = time.time()

            frame = ImageGrab.grab(all_screens=True)
            
            # 此处根据 model类别对获取的图像进行处理
            if self.model > self.MODEL.DEFAULT_MODEL:
                frame = self.executor.inference(frame, self.model)

            if frame is not None:
                frame = frame.resize(DEFAULT_SIZE)

                imgByteArr = io.BytesIO()
                frame.save(imgByteArr, format='Jpeg')
                imgByteArr = imgByteArr.getvalue()
                frame = bytes(imgByteArr) + b'\xff\xd9'

                frame_number += 1

                rtp_packet = RTPPacket(
                    payload_type=RTPPacket.TYPE.MJPEG,
                    sequence_number=frame_number,
                    timestamp=frame_number*self.DPS,
                    payload=frame
                )
                print(f"Sending packet #{frame_number}")
                print('Packet header:')
                rtp_packet.print_header()
                packet = rtp_packet.get_packet()
                self.send_rtp_packet(packet)

                t2 = time.time()
                if t2 - t1 > self.DPS:
                    sleep((self.DPS + t1 - t2) / 1000.)
        return


    def send_rtp_packet(self,packet):
        to_send = packet[:]
        while to_send:
            try:
                self.rtp_socket.sendto(to_send[:self.DEFAULT_CHUNK_SIZE], self.client_address)
            except socket.error as e:
                print(f"failed to send rtp packet: {e}")
                return
            # trim bytes sent
            to_send = to_send[self.DEFAULT_CHUNK_SIZE:]
        return
    
    def send_rtsp_response(self,sequence_number):
        response = RTSPPacket.build_response(sequence_number, self.SESSION_ID)
        self.rtsp_send(response.encode())
        print('Sent response to client.')
        return
    
    def rtsp_send(self, data):
        print(f"Sending to client: {repr(data)}")
        return self.rtsp_connection.send(data)
    
    def handle_rtsp_requests(self):
        print("Waiting for RTSP requests...")
        # main thread will be running here most of the time
        while True:
            packet = self.get_rtsp_packet()
            # assuming state will only ever be PAUSED or PLAYING at this point
            if packet.request_type == RTSPPacket.PLAY:
                if self.server_state == self.STATE.PLAYING:
                    print('Current state is already PLAYING.')
                    continue
                self.server_state = self.STATE.PLAYING
                print('State set to PLAYING.')
            elif packet.request_type == RTSPPacket.PAUSE:
                if self.server_state == self.STATE.PAUSED:
                    print('Current state is already PAUSED.')
                    continue
                self.server_state = self.STATE.PAUSED
                print('State set to PAUSED.')
            elif packet.request_type == RTSPPacket.TEARDOWN:
                print('Strat to Close the server')
                self.send_rtsp_response(packet.sequence_number)
                self.rtsp_connection.close()
                self.video_stream.close()
                self.rtp_socket.close()
                self.server_state = self.STATE.TEARDOWN
                # for simplicity's sake, caught on main_server
                raise ConnectionError('teardown requested')
            else:
                # will never happen, since exception is raised inside `parse_rtsp_request()`
                # raise InvalidRTSPRequest()
                pass
            self.send_rtsp_response(packet.sequence_number)
        return