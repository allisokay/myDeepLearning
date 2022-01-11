'''
@author：fc
@date：  2022/1/11
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# 交叉熵的理解：
交叉熵一般用于分类中loss的计算，给定类别集合class=[class1,class2,class3],每个类别对应的序号就是0,1,2
给定一个样本输入input,真实标签序号为1（class2）。模型输出的预测概率列表为:output=[0.1,0.2,0.7]，
计算公式：loss = -output[class]+ln(e^output[0]+e^output[1]+e^output[2])),模型预测结果为2（class3）,预测错误
      错误预测结果：  loss = -0.2+ln(e^(0.1)+e^(0.2)e^(0.7))，误差较大
      正确预测应该是：output=[0.1,0.7,0.2],loss = -0.7+ln(e^(0.1)+e^(0.2)e^(0.7)).-0.7<-0.2
"""

import torch
# 给定模型预测输出概率列表，
output = torch.tensor([0.1,0.2,0.7])  # 由结果可知，模型预测序号为2（class3）,而真实标签为1
output = torch.reshape(output,(1,3))  # 交叉熵损失中shape为batch_size,class_nums,样本数量为1个，有三种类别
rel_output = torch.tensor([0.1,0.7,0.2])
rel_output = torch.reshape(rel_output,(1,3))
target = torch.tensor([1])
# 计算交叉熵损失
loss_fun = torch.nn.CrossEntropyLoss()
loss = loss_fun(output,target)
less_loss = loss_fun(rel_output,target)
print(f"high loss = { loss},lower loss={less_loss}")