# -*- coding: utf-8 -*-
"""模块注释
生成一个简单的卷积神经网络：
1. 输入图像的大小为32x32x3（RGB图像）。
2. 第一层是卷积层1，使用5x5的过滤器、步长为1和6个过滤器。输出大小为28x28x6。
3. 接下来是池化层1，使用2x2的最大池化，步长为2。输出大小为14x14x6。
4. 然后是卷积层2，使用5x5的过滤器、步长为1和10个过滤器。输出大小为10x10x10。
5. 再进行池化层2，使用2x2的最大池化，步长为2。输出大小为5x5x10。
6. 接下来是卷积层3，使用5x5的过滤器、步长为1和16个过滤器。输出大小为1x1x400。
7. 将池化层2的输出展开成一个400x1的向量。
8. 然后是全连接层4，具有120个单元。
9. 最后一层是全连接层5，具有84个单元。
10. 最终输出层是具有10个单元的Softmax层，用于手写数字识别任务。

Author: Lei Wang
Date: April 24, 2024
"""
__author__ = "王磊"
__copyright__ = "Copyright 2023 四川科莫生医疗科技有限公司"
__credits__ = ["王磊"]
__maintainer__ = "王磊"
__email__ = "lei.wang@kemoshen.com"
__version__ = "0.0.1"
__status__ = "Development"


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层1: 输入通道3, 输出通道6, 卷积核大小5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 池化层1: 最大池化，窗口大小2x2，步长2
        self.pool1 = nn.MaxPool2d(2, 2)
        # 卷积层2: 输入通道6, 输出通道10, 卷积核大小5x5
        self.conv2 = nn.Conv2d(6, 10, 5)
        # 池化层2: 最大池化，窗口大小2x2，步长2
        self.pool2 = nn.MaxPool2d(2, 2)
        # 卷积层3: 输入通道10, 输出通道16, 卷积核大小5x5
        self.conv3 = nn.Conv2d(10, 16, 5)
        # 全连接层4: 输入尺寸400 (因为5x5x16=400), 输出尺寸120
        self.fc1 = nn.Linear(400, 120)
        # 全连接层5: 输入尺寸120, 输出尺寸84
        self.fc2 = nn.Linear(120, 84)
        # 输出层: 输入尺寸84, 输出尺寸10
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 通过卷积层1
        x = F.relu(self.conv1(x))
        # 通过池化层1
        x = self.pool1(x)
        # 通过卷积层2
        x = F.relu(self.conv2(x))
        # 通过池化层2
        x = self.pool2(x)
        # 通过卷积层3
        x = F.relu(self.conv2(x))
        # 展平
        x = x.view(-1, 400)
        # 通过全连接层4
        x = F.relu(self.fc1(x))
        # 通过全连接层5
        x = F.relu(self.fc2(x))
        # 通过输出层
        x = self.fc3(x)
        # 应用Softmax
        x = F.log_softmax(x, dim=1)
        return x

# 创建模型实例
model = SimpleCNN()
print(model)
