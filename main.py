import torch
from IPython import display
from matplotlib import pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np
import random
import sys
import os
import d2lzh_pytorch as d2l

# def use_svg_display():
#     # 用矢量图显示
#     matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
#
#
# def set_figsize(figsize=(7, 5)):
#     use_svg_display()
#     # 设置图的尺寸
#     plt.rcParams['figure.figsize'] = figsize

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

d2l.set_figsize(figsize=(7, 5))
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)

plt.show()
