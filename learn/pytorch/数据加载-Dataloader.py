'''
@author：fc
@date：  2022/1/10
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# torch.utils.data中的Dataloader的使用
"""
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter
import torchvision

train_set = torchvision.datasets.CIFAR10(root="G:/projects/PycharmProjects/Dataset/general/",train=True,transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(dataset=train_set,batch_size=64,shuffle=False,num_workers=0,drop_last=False)
writer = SummaryWriter("../../log/数据加载-Dataloader/")
for epoch in range(2):
    step = 0
    for data in train_loader:
        img,label = data
        writer.add_images("epoch:{}".format(epoch),img,step)
        step += 1
        print(f"图片对应的标签值:{label}")

writer.close()