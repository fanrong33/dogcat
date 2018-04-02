import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义神经网络模型
# 定义网络时，需要继承nn.Module，并实现它的forward方法，把网络中具有可学习参数的层放在构造函数__init__中。
# 如果某一层(如ReLU)不具有可学习的参数，则，既可以放在构造函数中，也可以不放，但建议不放在其中，而在forward中使用nn.functional代替。
class LeNet(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(LeNet, self).__init__()

        # 卷积层  '1'表示输入图片为单通道，'6'表示输出通道数，'5'表示卷积核为5*5
        # input shape (1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)  # output shape (16, 28, 28)
        # 卷积层
        # input shape (16, 14, 14)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)  # output shape (32, 14, 14)
        # 仿射层/全连接层，y = Wx + b
        self.fc1 = nn.Linear(32*7*7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2) # fully connected layer, output 10 classes

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # choose max value in 2x2 area, output shape (16, 14, 14)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape, '-1'表示自适应
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output