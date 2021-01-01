import torch
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import onnxruntime

import pyaudio
import wave

class AudioStream:
    def __init__(self, device_id, use_as_input, chunk_size):
        self.p = pyaudio.PyAudio()

        self.device_id = device_id
        device_info = self.p.get_device_info_by_index(self.device_id)
        self.is_input = device_info["maxInputChannels"] > 0
        self.is_wasapi = (self.p.get_host_api_info_by_index(device_info["hostApi"])["name"]).find("WASAPI") != -1
        self.useloopback = False
        if not self.is_input:
            assert self.is_wasapi, 'Selection is onput and does not support loopback mode'
            self.useloopback = True

        self.chunk_size = chunk_size
        self.rate = int(device_info["defaultSampleRate"])
        self.channels = device_info["maxInputChannels"] if (
                device_info["maxOutputChannels"] < device_info["maxInputChannels"]) else device_info["maxOutputChannels"]

        self.interval = self.chunk_size / self.rate / self.channels

        if use_as_input:
            self.stream = self.p.open(format=pyaudio.paInt16,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True,
                                 input_device_index=self.device_id,
                                 frames_per_buffer=self.chunk_size,
                                 as_loopback=self.useloopback)
        else:
            self.stream = self.p.open(format=pyaudio.paInt16,
                                 channels=self.channels,
                                 rate=self.rate,
                                 output=True)
        # for test
        self.wf = None

    def read_dummy(self, path):
        if self.wf is None:
            self.wf = wave.open(path, 'rb')
        data = self.wf.readframes(self.chunk_size)
        if len(data) == 0:
            self.wf = wave.open('out.wav', 'rb')
            data = self.wf.readframes(self.chunk_size)
        return data

    def read(self):
        """
        read audio bytes from an input stream into memory.
        Returns:
            bytes
        """
        return self.stream.read(self.chunk_size)

    def write(self, data):
        """
        write audio bytes to an output stream.
        Returns:
            None
        """
        self.stream.write(data)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def info():
        p = pyaudio.PyAudio()
        print("Available devices:\n")
        for i in range(0, p.get_device_count()):
            info = p.get_device_info_by_index(i)
            print(str(info["index"]) + ": \t %s \n \t %s \n" %
                  (info["name"], p.get_host_api_info_by_index(info["hostApi"])["name"]))

if __name__ == '__main__':
    AudioStream.info()

class Executor(object):
    def __init__(self):
        self.path = None
        self.model = None
        self.ort_session = None
        self.mtype = 'pt'

        # settings for image process
        self.cnn_normalization_mean = [0.485, 0.456, 0.406]
        self.cnn_normalization_std = [0.229, 0.224, 0.225]
        self.tensor_normalizer = transforms.Normalize(mean=self.cnn_normalization_mean, std=self.cnn_normalization_std)

    def load_model(self, paths, mtype='pt'):
        """
        load models into memory.
        Args:
            paths: dict {int: path-like str}
                model path with specfic key.
            mtype: 'pt' | 'onnx'
                model type.
        Returns:
            None
        """
        self.paths = paths
        self.mtype = mtype
        if self.mtype == 'pt':
            self.model = torch.load(self.path)
        elif self.mtype == 'onnx':
            self.ort_session_dict = {}
            for k, v in paths.items():
                self.ort_session_dict[k] = onnxruntime.InferenceSession(v)

    def inference(self, img, key):
        """
        model inference.
        Args:
            img: PIL
                input image.
            key: int
                specify a model path.
        Returns:
            img: PIL
        """
        # preprocess
        raw_size = img.width, img.height
        img = self.preprocess_image(img)

        # run model
        if self.mtype == 'pt':
            assert self.model is not None
            img = img.to('cuda:0')
            img = self.model(img)
        elif self.mtype == 'onnx':
            assert self.ort_session_dict is not None
            ort_inputs = {self.ort_session_dict[key].get_inputs()[0].name: img.numpy()}
            img = self.ort_session_dict[key].run(None, ort_inputs)[0]

        # post process
        img = self.recover_image(img, raw_size=raw_size)
        img = Image.fromarray(img)
        return img

    def scale_image(self, image, size, keep_ratio=True):
        """
        Args:
            image: PIL | [h, w, c]
                input image.
            size: int | (w, h)
                target size.
        Returns:
            img: PIL
                resized image.
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        ratio = min(size[0]/image.width, size[1]/image.height) if keep_ratio \
            else (size[0]/image.width, size[1]/image.height)
        if not isinstance(ratio, tuple):
            ratio = (ratio, ratio)

        if ratio[0] != 1 or ratio[1] != 1:
            image = image.resize((round(image.width * ratio[0]), round(image.height * ratio[1])),
                                 Image.NEAREST)
        return image

    def preprocess_image(self, image):
        """
        Args:
            image: PIL | [h, w, c]
                input image.
        Returns:
            img: [1, c, h, w]
                image tensor.
        """
        image = self.scale_image(image, (256, 256), keep_ratio=False)

        t = transforms.Compose([
            transforms.ToTensor(),
            self.tensor_normalizer,
        ])

        return t(image).unsqueeze(0)

    def recover_image(self, image, raw_size=None):
        """
        recover raw image from normalized tensor image.
        Args:
            image: [1, c, h, w]
                input image.
            raw_size: (w, h)
        Returns:
            img: [rh, rw, c]
                recovered image.
        """
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        image = image * np.array(self.cnn_normalization_std).reshape((1, 3, 1, 1)) + \
                np.array(self.cnn_normalization_mean).reshape((1, 3, 1, 1))

        image = (image.transpose(0, 2, 3, 1) * 255.).clip(0, 255).astype(np.uint8)[0]

        if raw_size is not None:
            image = self.scale_image(image, raw_size, keep_ratio=False)
            image = np.array(image, dtype=np.uint8)

        return image
