# encoding: utf-8
""" 评估训练的模型
"""

import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import requests

from datasets import DOGCAT
from models import LeNet

torch.manual_seed(1)


# Load Datasets
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(28),  # 缩放图片(Image), 保持长宽比不变，最短边为224像素
    torchvision.transforms.CenterCrop(28),  # 从图片中间切出224*224的图片
    torchvision.transforms.ToTensor(),  # 将图片(Image)转成Tensor, 归一化至[0, 1]
    torchvision.transforms.Normalize(mean=[.5,.5,.5], std=[.5, .5, .5])  # 标准化至[-1, 1], 规定均值和标准差
])


net = LeNet()
print(net)


net.load_state_dict(torch.load('saves/dogcat_lenet_params.pkl'))



url = 'https://ss1.baidu.com/6ONXsjip0QIZ8tyhnq/it/u=1876203052,395878551&fm=5'
r = requests.get(url)
im = Image.open(BytesIO(r.content))
# im = Image.open('123.jpeg')
# img.show()

image = transform(im)
print(image.size())
''' torch.Size([3, 28, 28]) '''

img = image.numpy()
img = np.transpose(img, (1,2,0))

plt.imshow(img, cmap='gray')
# plt.title('%s' % test_dataset.classes[label])
plt.show()



images = Variable(torch.unsqueeze(image, 0))
outputs = net(images)
_, predicted = torch.max(outputs.data, 1)
print(predicted)


