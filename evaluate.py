# encoding: utf-8
""" 评估训练的模型
"""

import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision

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

test_dataset = DOGCAT(
    root='../Pytorch-Tutorial/datasets/dogcat_2', 
    train=False,
    transform=transform
)

test_loader = Data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, num_workers=2)



net = LeNet()
print(net)


if os.path.isfile('saves/dogcat_lenet_params.pkl'):
    net.load_state_dict(torch.load('saves/dogcat_lenet_params.pkl'))
else:
    print("dogcat_lenet_params.pkl don't exists.")
    exit()


# Test the Model
total = 0
correct = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += sum(predicted == labels)
    
print('Test Accuracy of the model on the %d test images: %d %%' % (total, correct / total * 100)) 


