'''
@author：fc
@date：  2022/1/11
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# torch中的全连接层
"""
import torch
import torchvision
from torch.utils.data import DataLoader


class network(torch.nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.linear = torch.nn.Linear(196608,10)

    def forward(self,input):
        return self.linear(input)


if __name__ == '__main__':
    cifar10_test_dataset = torchvision.datasets.CIFAR10(root="G:/projects/PycharmProjects/Dataset/general/",train=False,transform=torchvision.transforms.ToTensor(),download=False)
    cifar_test_loader = DataLoader(cifar10_test_dataset,batch_size=64,shuffle=False,num_workers=0,drop_last=True)  # 不丢掉剩余的会报错
    for data in cifar_test_loader:
        imgs,labels = data
        print(f"图片形状{imgs.shape}")
        model = network()
        imgs_reshape = torch.reshape(imgs,[1,1,1,-1])
        imgs_flatten = torch.flatten(imgs)
        output_reshpe = model(imgs_reshape)
        output_flatten = model(imgs_flatten)
        print(f"变形输出：{output_reshpe.shape}，展平输出：{output_flatten.shape}")