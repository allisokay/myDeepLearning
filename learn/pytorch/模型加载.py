'''
@author：fc
@date：  2022/1/13
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
#
"""
import torch
from 模型的保存和加载 import *

# 加载以方式1（保存模型结构和参数）保存的模型，需要模型定义类
model = torch.load("../../model_saver/model_method1.pth")
manual_input = torch.normal(0, 1, [64, 3, 32, 32])
output = model(manual_input)
print(output)

del model
# 加载以方式2（保存模型参数）保存的模型，需要模型定义类和重新赋值再加载参数
model = network()
model.load_state_dict(torch.load("../../model_saver/model_method2.pth"))
manual_input = torch.normal(0, 1, [64, 3, 32, 32])
output = model(manual_input)
print(output)