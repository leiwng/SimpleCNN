# -*- coding: utf-8 -*-
"""模块注释

2. 设置损失函数和优化器
选择合适的损失函数和优化器来训织网络。

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

import torch.optim as optim

# 创建一个模型实例
model = SimpleCNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
