import torch
import torchvision.transforms as transforms

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import onnxruntime

from multiprocessing import Process, Queue
import time

cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]
tensor_normalizer = transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std)
epsilon = 1e-5

class Executor(object):
    def __init__(self, async=False):
        self.path = None
        self.model = None
        self.ort_session = None
        self.mtype = 'pt'
        self.rq = None
        self.sq = None
        self.require_buffering = True

        if async:
            self.start_pipeline()

    def load_model(self, paths, mtype='pt'):
        self.paths = paths
        self.mtype = mtype
        if self.mtype == 'pt':
            self.model = torch.load(self.path)
        elif self.mtype == 'onnx':
            self.ort_session_dict = {}
            for k, v in paths.items():
                self.ort_session_dict[k] = onnxruntime.InferenceSession(v)
            # self.ort_session = onnxruntime.InferenceSession(path)

    def start_pipeline(self):
        self.rq = Queue(60)
        self.sq = Queue(60)
        p = Process(target=self._inference_async, args=(10, self.rq, self.sq,))
        p.start()

    def put(self, img):
        self.rq.put(img)

    def get(self):
        return None if self.sq.empty() else self.sq.get()

    def _inference_async(self, batch_size, rq, sq):
        self.load_model('pretrained/transform_net_31_418_b10.onnx', mtype='onnx')

        while True:
            if self.require_buffering:
                if rq.qsize() < 20:
                    time.sleep(0.03)
                    continue
                else:
                    self.require_buffering = False

            imgs = []
            for i in range(batch_size):
                img = rq.get()
                # preprocess
                img = preprocess_image(img, target_width=256)
                imgs.append(img.numpy())

            imgs = np.concatenate(imgs, axis=0) # np.array(imgs, dtype=np.float32)
            ort_inputs = {self.ort_session.get_inputs()[0].name: imgs}
            imgs = self.ort_session.run(None, ort_inputs)[0]

            for img in imgs:
                # post process
                img = recover_image(img, raw_size=(640, 400))
                img = Image.fromarray(img)
                sq.put(img)

    def inference(self, img, key):
        # preprocess
        raw_size = img.width, img.height
        img = preprocess_image(img, target_width=256)

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
        img = recover_image(img, raw_size=raw_size)
        img = Image.fromarray(img)
        return img

def scale_image(image, size, keep_ratio=True):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    padding = False

    ratio = min(size[0]/image.width, size[1]/image.height) if keep_ratio \
        else (size[0]/image.width, size[1]/image.height)
    if not isinstance(ratio, tuple):
        ratio = (ratio, ratio)

    if ratio[0] != 1 or ratio[1] != 1:
        image = image.resize((round(image.width * ratio[0]), round(image.height * ratio[1])),
                             Image.NEAREST)

    if padding:
        image = np.array(image)
        ph = (size[1] - image.shape[0]) // 2
        pw = size[0]
        print(ph, pw, image.shape[2])
        padding_u = np.zeros((ph, pw, image.shape[2]), dtype=np.uint8)
        padding_d = np.zeros((size[1]-ph-image.shape[0], pw, image.shape[2]), dtype=np.uint8)
        image = np.concatenate([padding_u, image, padding_d], axis=0)
        image = Image.fromarray(image)

        return image, padding_u.shape[0], padding_d.shape[0]

    return image

def preprocess_image(image, target_width=None):
    """输入 PIL.Image 对象，输出标准化后的四维 tensor"""
    image = scale_image(image, (256, 256), keep_ratio=False)

    if target_width:
        t = transforms.Compose([
            transforms.ToTensor(),
            tensor_normalizer
        ])
    else:
        t = transforms.Compose([
            transforms.ToTensor(),
            tensor_normalizer,
        ])

    return t(image).unsqueeze(0)

def recover_image(image, padding=None, raw_size=None):
    """输入 [b, c, h, w], 输出 0~255 范围的三维 numpy 矩阵，RGB 顺序"""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = image * np.array(cnn_normalization_std).reshape((1, 3, 1, 1)) + \
            np.array(cnn_normalization_mean).reshape((1, 3, 1, 1))

    image = (image.transpose(0, 2, 3, 1) * 255.).clip(0, 255).astype(np.uint8)[0]

    if padding is not None:
        image = image[:, padding[0]:-1*padding[1], :]

    if raw_size is not None:
        image = scale_image(image, raw_size, keep_ratio=False)
        image = np.array(image, dtype=np.uint8)

    return image


def _preprocess_image(image, target_width=None):
    """输入 PIL.Image 对象，输出标准化后的四维 tensor"""
    if target_width:
        t = transforms.Compose([
            transforms.RandomResizedCrop(target_width, scale=(256 / 480, 1.), ratio=(1., 1.)),
            transforms.ToTensor(),
            tensor_normalizer
        ])
    else:
        t = transforms.Compose([
            transforms.ToTensor(), 
            tensor_normalizer, 
        ])
    return t(image).unsqueeze(0)


def image_to_tensor(image, target_width=None):
    """输入 OpenCV 图像，范围 0~255，BGR 顺序，输出标准化后的四维 tensor"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return preprocess_image(image, target_width)


def read_image(path, target_width=None):
    """输入图像路径，输出标准化后的四维 tensor"""
    image = Image.open(path)
    return preprocess_image(image, target_width)


def _recover_image(tensor):
    """输入 GPU 上的四维 tensor，输出 0~255 范围的三维 numpy 矩阵，RGB 顺序"""
    image = tensor.detach().cpu().numpy()
    image = image * np.array(cnn_normalization_std).reshape((1, 3, 1, 1)) + \
    np.array(cnn_normalization_mean).reshape((1, 3, 1, 1))
    return (image.transpose(0, 2, 3, 1) * 255.).clip(0, 255).astype(np.uint8)[0]


def recover_tensor(tensor):
    m = torch.tensor(cnn_normalization_mean).view(1, 3, 1, 1).to(tensor.device)
    s = torch.tensor(cnn_normalization_std).view(1, 3, 1, 1).to(tensor.device)
    tensor = tensor * s + m
    return tensor.clamp(0, 1)


def imshow(tensor, title=None):
    """输入 GPU 上的四维 tensor，然后绘制该图像"""
    image = recover_image(tensor)
    print(image.shape)
    plt.imshow(image)
    if title is not None:
        plt.title(title)


def mean_std(features):
    """输入 VGG16 计算的四个特征，输出每张特征图的均值和标准差，长度为1920"""
    mean_std_features = []
    for x in features:
        x = x.view(*x.shape[:2], -1)
        x = torch.cat([x.mean(-1), torch.sqrt(x.var(-1) + epsilon)], dim=-1)
        n = x.shape[0]
        x2 = x.view(n, 2, -1).transpose(2, 1).contiguous().view(n, -1) # 【mean, ..., std, ...] to [mean, std, ...]
        mean_std_features.append(x2)
    mean_std_features = torch.cat(mean_std_features, dim=-1)
    return mean_std_features


class Smooth:
    # 对输入的数据进行滑动平均
    def __init__(self, windowsize=100):
        self.window_size = windowsize
        self.data = np.zeros((self.window_size, 1), dtype=np.float32)
        self.index = 0
    
    def __iadd__(self, x):
        if self.index == 0:
            self.data[:] = x
        self.data[self.index % self.window_size] = x
        self.index += 1
        return self
    
    def __float__(self):
        return float(self.data.mean())
    
    def __format__(self, f):
        return self.__float__().__format__(f)