import socket
from lib.video_stream import VideoStream
from lib.rtsp_packet import RTSPPacket
from lib.rtp_packet import RTPPacket
from config import Config
from typing import Union,List
from PIL import Image
from threading import Thread
from io import BytesIO
from time import sleep

import collections
from utils import AudioStream
from lib.rtp_packet import InvalidPacketException

class Client:
    DEFAULT_CHUNK_SIZE = Config.CHUNK_SIZE
    DEFAULT_RECV_DELAY = Config.RECV_DELAY

    DEFAULT_HOST = Config.HOST
    LOCAL_HOST = Config.LOCAL_HOST
    RTP_TIMEOUT = Config.TIMEOUT/20
    RTSP_TIMEOUT = Config.TIMEOUT

    PACKET_HEADER_LENGTH = Config.PACKET_HEADER_LENGTH

    def __init__(self,file_path,host_ip,host_port,rtp_port):
        self.rtsp_connection: Union[None, socket.socket] = None
        self.rtp_socket: Union[None, socket.socket] = None
        self.rtp_receive_thread: Union[None, Thread] = None
        self.frame_buffer = collections.deque(maxlen=30)
        self.current_sequence_number = 0
        self.audio_stream = AudioStream(device_id=5, use_as_input=False, chunk_size=2048)
        self.session_id = ''

        self.current_frame_cnt = -1

        self.is_rtsp_connected = False
        self.is_receiving_rtp = False

        self.file_path = file_path
        self.remote_host_address = host_ip
        self.remote_host_port = host_port
        self.rtp_port = rtp_port

    def get_next_frame(self):
        if self.frame_buffer:
            self.current_frame_cnt +=1
            return self.frame_buffer.popleft(), self.current_frame_cnt
        return None
    
    def decode_packet(self, packet):
        img = packet.payload
        img = img[:len(img)-2]
        frame = Image.open(BytesIO(img))
        return frame
    
    def recv_rtp_packet(self, size=DEFAULT_CHUNK_SIZE, type='video'):
        recv = bytes()
        while True:
            try:
                if type == 'video':
                    recv += self._rtp_socket.recv(size)
                    if recv.endswith(VideoStream.JPEG_EOF):
                        break
                elif type == 'audio':
                    recv += self._rtp_audio_socket.recv(size)
                    if recv.endswith(VideoStream.JPEG_EOF):
                        break
            except socket.timeout:
                continue
        try:
            packet = RTPPacket.from_packet(recv)
        except InvalidPacketException:
            pass
        return packet

    def start_rtp_receive_thread(self):
        self._rtp_receive_thread = Thread(target=self.handle_video_receive)
        self._rtp_receive_thread.setDaemon(True)
        self._rtp_receive_thread.start()

        self._rtp_audio_receive_thread = Thread(target=self.handle_audio_receive)
        self._rtp_audio_receive_thread.setDaemon(True)
        self._rtp_audio_receive_thread.start()

    def handle_audio_receive(self):
        self._rtp_audio_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rtp_audio_socket.bind((self.LOCAL_HOST, self.rtp_port+1))
        self._rtp_audio_socket.settimeout(self.RTP_TIMEOUT / 1000.)
        while True:
            if not self.is_receiving_rtp:
                sleep(self.RTP_TIMEOUT/1000.)  # diminish cpu hogging
                continue
            packet = self.recv_rtp_packet(type='audio')
            if packet.payload_type == RTPPacket.TYPE.AUDIO:
                frame = packet.payload
                self.audio_stream.write(frame)

    def handle_video_receive(self):
        self._rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rtp_socket.bind((self.LOCAL_HOST, self.rtp_port))
        self._rtp_socket.settimeout(self.RTP_TIMEOUT / 1000.)
        while True:
            if not self.is_receiving_rtp:
                sleep(self.RTP_TIMEOUT/1000.)  # diminish cpu hogging
                continue
            packet = self.recv_rtp_packet()
            if packet.payload_type == RTPPacket.TYPE.MJPEG:
                frame = self.decode_packet(packet)
                self.frame_buffer.append(frame)


    def establish_rtsp_connection(self):
        if self.is_rtsp_connected:
            print('RTSP is already connected.')
            return
        print(f"Connecting to {self.remote_host_address}:{self.remote_host_port}...")
        self.rtsp_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rtsp_connection.connect((self.remote_host_address, self.remote_host_port))
        self.rtsp_connection.settimeout(self.RTSP_TIMEOUT / 1000.)
        self.is_rtsp_connected = True

    def close_rtsp_connection(self):
        if not self.is_rtsp_connected:
            print('RTSP is not connected.')
            return
        self.rtsp_connection.close()
        self.is_rtsp_connected = False

    def send_request(self, request_type=RTSPPacket.INVALID) -> RTSPPacket:
        if not self.is_rtsp_connected:
            raise Exception('rtsp connection not established. run `setup_rtsp_connection()`')
        request = RTSPPacket(
            request_type,
            self.file_path,
            self.current_sequence_number,
            self.rtp_port,
            self.session_id
        ).to_request()
        print(f"Sending request: {repr(request)}")
        self.rtsp_connection.send(request)
        self.current_sequence_number += 1
        return self.get_response()

    def send_setup_request(self) -> RTSPPacket:
        response = self.send_request(RTSPPacket.SETUP)
        self.start_rtp_receive_thread()
        self.session_id = response.session_id
        return response

    def send_play_request(self) -> RTSPPacket:
        response = self.send_request(RTSPPacket.PLAY)
        self.is_receiving_rtp = True
        return response

    def send_pause_request(self) -> RTSPPacket:
        response = self.send_request(RTSPPacket.PAUSE)
        self.is_receiving_rtp = False
        return response

    def send_teardown_request(self) -> RTSPPacket:
        response = self.send_request(RTSPPacket.TEARDOWN)
        self.is_receiving_rtp = False
        self.is_rtsp_connected = False
        return response

    def get_response(self, size=DEFAULT_CHUNK_SIZE) -> RTSPPacket:
        rcv = None
        while True:
            try:
                rcv = self.rtsp_connection.recv(size)
                break
            except socket.timeout:
                continue
        print(f"Received from server: {repr(rcv)}")
        response = RTSPPacket.from_response(rcv)
        return response
