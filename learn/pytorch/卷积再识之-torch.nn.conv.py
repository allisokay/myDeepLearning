'''
@author：fc
@date：  2022/1/10
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# torch.nn.conv2d的初次使用
"""
import warnings
warnings.filterwarnings("ignore")
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def get_data():
    cifar10_test_dataset = torchvision.datasets.CIFAR10(root="G:/projects/PycharmProjects/Dataset/general/",
                                                        train=False, transform=torchvision.transforms.ToTensor(),
                                                        download=True)
    test_loader = DataLoader(dataset=cifar10_test_dataset, batch_size=64, shuffle=False, num_workers=0,
                              drop_last=False)
    return test_loader


class network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # torch.nn.Conv2d的卷积操作，参数为:输入通道数，输入通道数，卷积核大小为3*3
        self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3)

    def forward(self,x):
        # 我知道它调用了父类的__call__方法，但我中觉得我的代码中没在init函数中接收，可它确实被用了，没有torch.nn.functinal.conv2d直接输入样本张量明了。玄学
        return self.conv1(x)


if __name__ == '__main__':
    writer = SummaryWriter("../../log/toch.nn中的Conv2d")
    data_loader = get_data()
    model = network()
    for epoch in range(1):
        step = 0
        # data的大小取决于Dataloader中的batch_size
        for data in data_loader:
            step +=1
            imgs,labels = data
            writer.add_images(f"input_img",imgs,step)
            # 返回的结果中shape是：torch.Size([16,6,30,30])
            output = model(imgs)
            # 将output转换为图片格式以便tensorboard查看
            output = torch.reshape(output,[-1,3,30,30])
            writer.add_images(f"out_img",output,step)
