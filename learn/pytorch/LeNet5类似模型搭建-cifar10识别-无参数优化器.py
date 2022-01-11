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
    # 手工样本模型检查
    model = network()
    print(model)
    manual_input = torch.ones([64,3,32,32])
    out = model(manual_input)
    print(out.shape)
    writer = SummaryWriter("../../log/writer.add_graph图形查看")
    writer.add_graph(model,manual_input)
    del out
    del model

    # cifar10数据集的使用
    cifar10_test_dataset = torchvision.datasets.CIFAR10(root="G:/projects/PycharmProjects/Dataset/general/",train=False,transform=torchvision.transforms.ToTensor(),download=False)
    cifar10_loader = DataLoader(dataset=cifar10_test_dataset,batch_size=1,shuffle=True)

    model = network()
    cross_loss_func = torch.nn.CrossEntropyLoss()
    for data in cifar10_loader:
        x ,y = data
        print(f"该样本的序号为：{y}，标签值是{cifar10_test_dataset.classes[y]}")
        writer.add_images("输入图片查看",x, global_step=0)
        output = model(x)
        print(f"预测序号概率列表:{output}")
        writer.add_graph(model,x)
        output = torch.reshape(output,(1,10)) # 样本数量一个，类别10类
        loss=cross_loss_func(output,y)
        print(f"交叉熵损失为：{loss}")
        break
    writer.close()
