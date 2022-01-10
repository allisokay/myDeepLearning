'''
@author：fc
@date：  2022/1/10
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# torch中的池化操作
"""
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader

class network(torch.nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.maxPool = torch.nn.MaxPool2d(kernel_size=3,
                                          ceil_mode=True)  # 个人：torch中，stride(移动的步长会和kernel大小一样),ceil_mode:是否填充边缘

    def forward(self, input):
        return self.maxPool(input)

if __name__ == '__main__':
    manual_input = [
        [1, 2, 0, 3, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [5, 2, 3, 1, 1],
        [2, 1, 0, 1, 1]
    ]
    manual_tensor_input = torch.tensor(manual_input, dtype=torch.float32)
    manual_tensor_input = torch.reshape(manual_tensor_input,[-1,1,5,5])
    model= network()
    manual_output = model(manual_tensor_input)
    print(f"手工样本池化后结果为{manual_output}")

    cifar_test_dataset = torchvision.datasets.CIFAR10(root="G:/projects/PycharmProjects/Dataset/general/",train=False,transform=torchvision.transforms.ToTensor(),download=True)
    data_loader = DataLoader(dataset=cifar_test_dataset,batch_size=64,shuffle=False,num_workers=0,drop_last=False)

    step = 0
    writer = SummaryWriter("../../log/toch.nn中池化")
    del model
    model = network()
    for data in data_loader:
        imgs,labels = data
        step +=1
        writer.add_images("cifar原图",imgs,step)
        cifar_output = model(imgs)
        writer.add_images("torch中的池化后",cifar_output,step)
    writer.close()
