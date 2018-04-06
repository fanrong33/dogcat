# encoding: utf-8
""" 训练神经网络
"""

import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from datasets import DOGCAT
from models import LeNet

torch.manual_seed(1)


# Hyper Parameters
EPOCH = 10       # 训练
BATCH_SIZE = 50
LR = 0.001


# Load Datasets
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(28),  # 缩放图片(Image), 保持长宽比不变，最短边为224像素
    torchvision.transforms.CenterCrop(28),  # 从图片中间切出224*224的图片
    torchvision.transforms.ToTensor(),  # 将图片(Image)转成Tensor, 归一化至[0, 1]
    torchvision.transforms.Normalize(mean=[.5,.5,.5], std=[.5, .5, .5])  # 标准化至[-1, 1], 规定均值和标准差
])

train_dataset = DOGCAT(
    root='../Pytorch-Tutorial/datasets/dogcat_2/',
    train=True,
    transform=transform
)

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_dataset = DOGCAT(
    root='../Pytorch-Tutorial/datasets/dogcat_2', 
    train=False,
    transform=transform
)

test_loader = Data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=True, num_workers=2)

(test_images, test_labels) = iter(test_loader).next()
test_images = Variable(test_images)  # Tensor to Variable


# plot one exampel
print(len(train_dataset))
''' 17500 '''
(data, label) = train_dataset[100]
print(data.size())
''' torch.Size([3, 28, 28]) '''

img = data
img = img.numpy()
img = np.transpose(img, (1,2,0))

plt.imshow(img, cmap='gray')
plt.title('%s' % train_dataset.classes[label])
plt.show()



net = LeNet()
print(net)


if os.path.isfile('saves/dogcat_lenet_params.pkl'):
    net.load_state_dict(torch.load('saves/dogcat_lenet_params.pkl'))


# optimizer & Loss
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    for step, (images, labels) in enumerate(train_loader):

        images = Variable(images)
        labels = Variable(labels)
        print(images.data.numpy())
        print(labels.data.numpy())
        exit()
        
        optimizer.zero_grad()

        prediction = net(images)
        loss = loss_func(prediction, labels)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_prediction = net(test_images)
            _, predicted = torch.max(test_prediction.data, 1)
            test_accuracy = sum(predicted == test_labels)/float(test_labels.size(0))

            print('Epoch: %d, Step: %d, Training Loss: %.4f, Test Accuracy: %.3f' % 
                (epoch, step, loss.data[0], test_accuracy))

            torch.save(net.state_dict(), 'saves/dogcat_lenet_params.pkl')  # 只保存网络中的参数（速度快，占内存少）


