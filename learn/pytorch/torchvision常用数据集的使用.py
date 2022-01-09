'''
@author：fc
@date：  2022/1/9
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# pytorch中的torchvision对常用的数据集进行了封装、可以直接使用
"""
import torchvision
from torch.utils.tensorboard import SummaryWriter
import random
import warnings
warnings.filterwarnings("ignore")
trans_dataset = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="G:/projects/PycharmProjects/Dataset/general/", train=True,
                                         transform=trans_dataset, download=True)
test_set = torchvision.datasets.CIFAR10(root="G:/projects/PycharmProjects/Dataset/general/", train=False,
                                        transform=trans_dataset, download=False)

writer = SummaryWriter("../../log/torchvision常用数据集的使用")

for i in range(10):
    img, target = train_set[random.randint(0, 50000)]
    writer.add_image("tensorboard1",img,i)
    print(f"图片对应的标签为：{target}")