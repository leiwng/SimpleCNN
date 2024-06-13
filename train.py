# -*- coding: utf-8 -*-
"""模块注释
要训练上述定义的简单卷积神经网络（SimpleCNN），我们需要设置一个训练循环，其中包括数据加载、模型优化和损失计算。

3. 训练模型
循环遍历数据集，将数据输入到模型中，并进行优化。

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

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入；数据是[inputs, labels]
        inputs, labels = data

        # 清零梯度缓存
        optimizer.zero_grad()

        # 前向传播 + 反向传播 + 优化
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每2000个小批量打印一次
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running+loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
