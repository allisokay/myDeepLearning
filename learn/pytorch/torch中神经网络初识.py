'''
@author：fc
@date：  2022/1/10
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# torch中的神经网络初识
"""
import torch
from torch import nn,functional as F

class network(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        return input+1


if __name__ == '__main__':
    model = network()
    input = torch.tensor(1.0)
    output = model(input)
    print(f"搭建x+1的神经网络,输入{input}后得到输出{output}")