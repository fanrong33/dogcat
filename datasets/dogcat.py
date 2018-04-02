# encoding: utf-8
""" 自定义数据集 猫狗
"""

import torch
import torch.utils.data as Data
import torchvision

import os
from PIL import Image
import numpy as np
import random

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_image_file(filename):
    """ 检查一个文件是否为图片

    Args:
        filename (string): 一个文件的路径

    Returns:
        bool: True 如果文件后缀格式为图片格式
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def find_classes(dir):
    """ 根据目录返回对应的分类和索引
        ['cat', 'dog'], {'cat': 0, 'dog': 1}
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])  # tuple()
                    images.append(item)
    return images


class DOGCAT(Data.Dataset):
    def __init__(self, root, transform=None, train=True):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        random.shuffle(imgs)

        # imgs = os.listdir(root)
        # 所有图片的绝对路径
        # 这里不实际加载图片，只是指定路径，当调用__getitem__时才会真正读图片
        # imgs = [os.path.join(root, img) for img in imgs]
        count = len(imgs)
        
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.train = train

        if self.train:
            self.imgs = imgs[:int(0.7*count)]
        else:
            self.imgs = imgs[int(0.7*count):]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, label = self.imgs[index]
        data = Image.open(path)

        # 这样做可以与其他数据集一样保持一致，返回一个 PIL Image
        # 注意：data返回的是原始数据的类型，比如图片为PILImage，而不是Tensor, Tensor化由transform处理

        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def __len__(self):
        return len(self.imgs)

