'''
@author：fc
@date：  2022/1/13
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# 我是有些抗拒图像模型的，但基本上所有讲解都是基于图像模型的，这里使用torchvision中的vgg16
"""
import warnings
warnings.filterwarnings("ignore")
import torchvision
import torch
from torch.utils.tensorboard import SummaryWriter

vgg16_true = torchvision.models.vgg16(pretrained=True)
print(f"vgg16已训练好参数的引入：{vgg16_true}")

vgg16_false = torchvision.models.vgg16(pretrained=False)
print(f"vgg16未选练好参数模型的引入：{vgg16_false}")

# vgg16添加线性层(添加第7层)
vgg16_true.classifier.add_module("7",torch.nn.Linear(1000,10))
print(f"vgg16在最后一层添加线性层后模型查看：{vgg16_true}")
# vgg16修改模型，第6层输出变为1024，刚才添加的层变为1024,16
vgg16_true.classifier[6] = torch.nn.Linear(4096,1024)
vgg16_true.classifier[7] = torch.nn.Linear(1024,10)
print(f"vgg16修改模型后查看：{vgg16_true}")
# 检查模型
writer = SummaryWriter("../../log/随便生成的图片(满足正太分布：均值0，方差1)查看")
manual_input = torch.normal(0,1,[1,3,32,32])
print(f"输入样本：\n{manual_input}")
writer.add_images("手动生成图片(满足正太分布：均值0，方差1)",manual_input,0)
writer.close()
output = vgg16_true(manual_input)
print(f"模型输出：\n{output}")

