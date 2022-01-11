'''
@author：fc
@date：  2022/1/11
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# 卷积+池化+卷积+池化+卷积+池化+Flatten+全连接
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore")


class network(torch.nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)  # padding=2使得卷积后图像的长宽不变
        self.maxPool1 = torch.nn.MaxPool2d(2)  # 图像长宽减半
        self.conv2 = torch.nn.Conv2d(32,32,kernel_size=5,padding=2)
        self.maxPool2 = torch.nn.MaxPool2d(2)
        self.conv3 = torch.nn.Conv2d(32,64,kernel_size=5,padding=2)
        self.maxPool3 = torch.nn.MaxPool2d(2)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(64*4*4,64)
        self.linear2 = torch.nn.Linear(64,10)

    def forward(self,input):
        res = self.conv1(input)
        res = self.maxPool1(res)
        res = self.conv2(res)
        res = self.maxPool2(res)
        res = self.conv3(res)
        res = self.maxPool3(res)
        res = self.flatten(res)
        res = self.linear1(res)
        res = self.linear2(res)
        return  res

if __name__ == '__main__':
    model = network()
    print(model)
    input = torch.ones([64,3,32,32])
    out = model(input)
    print(out.shape)

