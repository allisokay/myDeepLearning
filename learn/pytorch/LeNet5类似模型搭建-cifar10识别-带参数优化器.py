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
from torch.utils.data import DataLoader
import torchvision
import random


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter("../../log/writer.add_graph图形查看")

    # cifar10数据集的使用
    cifar10_train_dataset = torchvision.datasets.CIFAR10(root="G:/projects/PycharmProjects/Dataset/general/",train=True,transform=torchvision.transforms.ToTensor(),download=False)
    cifar10_loader = DataLoader(dataset=cifar10_train_dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

    model = network()
    model.to(device)
    cross_loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(),lr=0.01)

    for epoch in range(10):
        train_cross_epoch_loss = 0.0
        for data in cifar10_loader:
            x ,y = data
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            writer.add_graph(model,x)
            output = torch.reshape(output,[64,10]) # 样本数量64个，类别10类
            loss=cross_loss_func(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_cross_epoch_loss += loss
        print(f"epoch       ：{train_cross_epoch_loss}")
        writer.add_scalar("loss_epoch",train_cross_epoch_loss,epoch)

    writer.close()

    # 测试数据集加载
    cifar10_test_dataset = torchvision.datasets.CIFAR10(root="G:/projects/PycharmProjects/Dataset/general/",
                                                        train=False, transform=torchvision.transforms.ToTensor(),
                                                        download=False)
    cifar10_test_loader = DataLoader(dataset=cifar10_test_dataset, batch_size=100, shuffle=False)

    conut = 0
    for data in cifar10_test_loader:
        if conut<50: # 每个迭代取第五十个样本来预测
            conut += 1
            continue
        else:
            imgs,target = data
            out = model(imgs[conut])
            writer.add_images("样本图形", imgs[conut],conut)
            print(f"预测结果：{out}")
            print(f"该样本预测类别是：{cifar10_test_dataset.classes[torch.argmax(output)]}")
