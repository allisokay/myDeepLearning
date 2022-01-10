'''
@author：fc
@date：  2022/1/10
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# torch中的卷积初识，初识了几十次了我都不好意思了
"""
import torch
from torch.nn import functional as F
input = [
    [1,2,0,3,1],
    [0,1,2,3,1],
    [1,2,1,0,0],
    [5,2,3,1,1],
    [2,1,0,1,1]
]

# 卷积核
filters=[
    [1,2,1],
    [0,1,0],
    [2,1,0]
]
# 输入转张量然后变形
tensor_input = torch.tensor(input)
tensor_input = torch.reshape(tensor_input,[1,1,5,5])
# 卷积核转张量然后变形
tensor_filters = torch.tensor(filters)
tensor_filters = torch.reshape(tensor_filters,[1,1,3,3])

# 卷积操作：padding默认为0
output1 = F.conv2d(tensor_input,tensor_filters,stride=1)
print(f"卷积：padding默认为0时：\n \t{output1}")

# 卷积操作：padding默认为0,步长为2
output2 = F.conv2d(tensor_input,tensor_filters,stride=2)
print(f"卷积：padding默认为0,步长为2：\n \t{output2}")

# 卷积操作：padding设置为1
output3 = F.conv2d(tensor_input,tensor_filters,stride=1,padding=1)
print(f"卷积：padding设置为1时：\n \t{output3}")