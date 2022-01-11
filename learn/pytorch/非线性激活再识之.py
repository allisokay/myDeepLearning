'''
@author：fc
@date：  2022/1/11
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# torch中激活函数Relu、sigmoid初识
"""
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision

class network(torch.nn.Module):
    def __init__(self,):
        super(network, self).__init__()
        self.relu=torch.nn.ReLU() # 果然torch.nn中的函数都与输入无关，调用时才输入
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,input):
        return self.relu(input),self.sigmoid(input)


if __name__ == '__main__':
    manual_input = [
        [1,-0.5],
        [-1,3],
        [0,2]
    ]
    tensor_manual_input = torch.tensor(manual_input,dtype=torch.float32)
    tensor_manual_input = torch.reshape(tensor_manual_input,[-1,1,3,2])
    model = network()
    output_relu,output_sigmoid = model(tensor_manual_input)
    print(f"手动数据集激活结果,relu:{output_relu},sigmoid:{output_sigmoid}")
    del model

    cifar10_test_dataset = torchvision.datasets.CIFAR10(root="G:/projects/PycharmProjects/Dataset/general/",train=False,transform=torchvision.transforms.ToTensor(),download=True)
    cifar10_test_dataloader = DataLoader(dataset=cifar10_test_dataset,batch_size=25,shuffle=False,num_workers=0,drop_last=False)
    writer = SummaryWriter("../../log/torch.nn中的激活函数")
    step = 0
    model = network()
    for data in cifar10_test_dataloader:
        imgs,labels = data
        writer.add_images("原始图片查看",imgs,step)

        outputs_relu,outputs_sigmoid = model(imgs)
        writer.add_images("relu激活后图片查看：",outputs_relu,step)
        writer.add_images("sigmoid激活后图片查看：",outputs_sigmoid,step)
        step += 1
    writer.close()