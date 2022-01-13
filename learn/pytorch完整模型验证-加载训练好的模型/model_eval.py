"""
@author：fc
@date：  2022/1/13
@contact：675435108@qq.com
"""
# 从浏览器中找五张图片进行预测cifar10数据集训练的模型
from cifar10_model import *
import torch
from PIL import Image
import os
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


def get_raw_imgs():
    print("-----加载数据-----")
    base_path = "imgs"
    img_names = os.listdir(base_path)
    tensor_imgs = []
    transform = torchvision.transforms.Compose([
         torchvision.transforms.Resize((32,32)),
         torchvision.transforms.ToTensor()
    ])
    for path in img_names:
        img_path = os.path.join(base_path, path)
        img = Image.open(img_path)  # 有时候我也是猪脑子，这个路径要拼接上文件夹，我直接就引用了，然后还百度找问题。
        img = transform(img)
        tensor_imgs.append(img)
    print("-----加载完成-----")
    tensor_imgs = torch.stack(tensor_imgs,axis=0)
    return tensor_imgs


if __name__ == '__main__':
    cifar10_test_dataset = torchvision.datasets.CIFAR10(root="G:/projects/PycharmProjects/Dataset/general/", train=False,
                                                        transform=torchvision.transforms.ToTensor())
    classes = cifar10_test_dataset.classes
    test_dataset_loader = DataLoader(dataset=cifar10_test_dataset, batch_size=6,shuffle=True)
    model = torch.load("../../model_saver/cifar10.pth")
    # 对于cifar测试集的预测
    for data in test_dataset_loader:
        imgs , targets = data
        imgs = imgs.cuda()
        out = model(imgs)
        out_class_nums = out.argmax(1)
        i = 1
        for img in imgs:
            img = img.cpu().numpy()
            img = np.transpose(img,(1,2,0))  # 转变维度,将tensor的C、H、W转换为H、W、C
            plt.title(f"pred:{classes[out_class_nums[i-1]]},real:{classes[targets[i-1]]}")
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
            plt.show()
            i = i+1
        break
    del model
    model = torch.load("../../model_saver/cifar10.pth")
    # 浏览器导入图片的预测
    imgs_tensor = get_raw_imgs()
    imgs_tensor = imgs_tensor.cuda()
    out = model(imgs_tensor)
    i = 1
    for img in imgs_tensor:
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # 转变维度,将tensor的C、H、W转换为H、W、C
        plt.title(f"pred:{classes[out_class_nums[i - 1]]}")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        plt.show()
        i = i + 1

