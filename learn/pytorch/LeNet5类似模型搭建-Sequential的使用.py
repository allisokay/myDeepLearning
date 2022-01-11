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
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),  # padding=2使得卷积后图像的长宽不变
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, kernel_size=5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 64),
            torch.nn.Linear(64, 10)
        )

    def forward(self,input):
        return self.model(input)


if __name__ == '__main__':
    model = network()
    print(model)
    input = torch.ones([64,3,32,32])
    out = model(input)
    print(out.shape)
    writer = SummaryWriter("../../log/writer.add_graph图形查看")
    writer.add_graph(model,input)
