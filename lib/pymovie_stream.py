# encoding: utf-8
"""
@author:  zhaominyi
@contact: 20210240011@fudan.edu.cn
"""
from moviepy.editor import VideoFileClip
from PIL import Image,ImageTk
import numpy as np
import io
import os
class VideoStream:

    DEFAULT_FPS = 24
    DEFAULT_SIZE = (320*2,200*2)
    VIDEO_LENGTH = 500
    JPEG_EOF = b'\xff\xd9'
    # if it's present at the end of chunk,
    # it's the last chunk for current jpeg (end of frame)

    def __init__(self, file_path: str):
        # for simplicity, mjpeg is assumed to be on working directory
        self.video = VideoFileClip(file_path)
        self.video_iter = self.video.iter_frames(fps = self.DEFAULT_FPS)
        # frame number is zero-indexed
        # after first frame is sent, this is set to zero
        self.t_list =list(np.arange(0, self.video.duration, 1.0/self.DEFAULT_FPS))
        self.max_frame_number = len(self.t_list)
        self.VIDEO_LENGTH = self.max_frame_number
        self.current_frame_number = -1

    def close(self):
        self.video.close()

    def get_next_frame(self) -> bytes:
        # sample video file format is as follows:
        # - 5 digit integer `frame_length` written as 5 bytes, one for each digit (ascii)
        # - `frame_length` bytes follow, which represent the frame encoded as a JPEG
        # - repeat until EOF
        self.current_frame_number +=1
        if self.current_frame_number == self.max_frame_number-1:
            return None
        frame = self.video.get_frame(self.t_list[self.current_frame_number])
        frame = Image.fromarray(frame).resize(self.DEFAULT_SIZE)
        # frame.show()
        # os._exit(233)
        imgByteArr = io.BytesIO()
        frame.save(imgByteArr,format='Jpeg')
        imgByteArr = imgByteArr.getvalue()
        # print("IMGBYTE=",imgByteArr)
        # imgByteArr += self.JPEG_EOF
        # print("frame=",frame)
        # print(len(bytes(imgByteArr)),len(self.JPEG_EOF))
        # frame = ImageTk.PhotoImage(fr)
        return bytes(imgByteArr)+self.JPEG_EOF
