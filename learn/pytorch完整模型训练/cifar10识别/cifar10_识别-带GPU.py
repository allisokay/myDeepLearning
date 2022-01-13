'''
@author：fc
@date：  2022/1/13
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# 不使用gpu识别cifar10完整模型训练
"""
import warnings
warnings.filterwarnings("ignore")
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from cifar10_model import Network


def load_data(data_path="G:/projects/PycharmProjects/Dataset/general/"):
    print("------加载cifar10数据集------")
    cifar10_train_dataset =  torchvision.datasets.CIFAR10(root=data_path,train=True,transform=torchvision.transforms.ToTensor())
    cifar10_test_dataset =  torchvision.datasets.CIFAR10(root=data_path,train=False,transform=torchvision.transforms.ToTensor())
    print(f"训练集长度 :{len(cifar10_train_dataset)},测试集长度 :{len(cifar10_test_dataset)}")
    train_dataset_loader = DataLoader(dataset=cifar10_train_dataset,batch_size=36,shuffle=True,num_workers=0)
    test_dataset_loader = DataLoader(dataset=cifar10_test_dataset,batch_size=16)
    print("------数据加载完成------")
    return train_dataset_loader,test_dataset_loader,cifar10_train_dataset,cifar10_test_dataset



def train():
    writer = SummaryWriter("../../../log/pytorch完整模型训练")
    train_data, test_data,train_dataset,test_dataset = load_data()
    classes =  test_dataset.classes
    model = Network()
    model = model.cuda()
    train_step = 0  # 训练次数处
    test_step = 0   # 测试次数
    lr = 1e-2
    epochs = 20
    loss_func = torch.nn.CrossEntropyLoss()
    loss_func = loss_func.cuda()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    avg_loss_saver = []
    model.train()
    each_step_accuracy = 0
    max_step_accuracy= 0
    for i in range(epochs):
        print(f"------第{i+1}轮训练开始------")
        epoch_total_loss = 0
        count = 0
        epoch_steps = 0
        for data in train_data:
            train_step += 1
            epoch_steps += 1
            imgs,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            output = model(imgs)
            loss = loss_func(output,targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if train_step % 100 ==0:
                print(f"     第{train_step}次训练,loss:{loss.item()}")
            epoch_total_loss += loss.item()
            count += 1
        avg_loss = epoch_total_loss / count
        avg_loss_saver.append(avg_loss)
        print(print(f"------第{i+1}轮训练结束，训练：{epoch_steps}次，每次平均loss:{avg_loss}---"))
        writer.add_scalar("cifar10模型训练每轮平均loss",avg_loss,i)

        # 测试该轮训练优化情况
        model.eval()
        with torch.no_grad():
            break_count = 1
            for data  in test_data:
                imgs,targets = data
                imgs = imgs.cuda()
                targets = targets.cuda()
                writer.add_images("测试图片",imgs,test_step)
                out = model(imgs)
                test_loss = loss_func(out, targets)
                test_step += 1
                each_step_accuracy = (out.argmax(axis=1) == targets).sum() / len(imgs)
                if each_step_accuracy>max_step_accuracy:
                    max_step_accuracy = each_step_accuracy
                    torch.save(model,"../../../model_saver/cifar10.pth")

                if test_step % 100 == 0:
                    break_count += 1
                    print(f"------第{test_step}次测试，loss:{test_loss},准确率：{each_step_accuracy}")
                    writer.add_scalar("每100次测试的准确率",each_step_accuracy,test_step)
                    writer.add_scalar("每100次测试的loss",test_loss,test_step)
                    class_seq = torch.argmax(out, axis=1)
                    print(f"图片真实类别,     预测类别")
                    for j in range(len(imgs)):
                      print(f"   {classes[targets[j]]},    {classes[class_seq[j]]} ")
                    if break_count == 10:
                        break

    writer.close()


if __name__ == '__main__':
    train()