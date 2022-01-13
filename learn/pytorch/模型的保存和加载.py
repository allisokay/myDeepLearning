'''
@author：fc
@date：  2022/1/13
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# 模型的保存
"""

import torch


class network(torch.nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 9, kernel_size=3),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(9, 64, kernel_size=3),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 512, kernel_size=3),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 2 * 2, 10)
        )

    def forward(self, input):
        return self.model(input)


if __name__ == '__main__':
    model = network()
    manual_input = torch.normal(0, 1, [64, 3, 32, 32])
    output = model(manual_input)
    print(output.shape)
    torch.save(model, "../../model_saver/model_method1.pth")  # 保存模型结构和参数，这种方式模型加载时，所在文件中必须存在模型定义类，但不用重新赋值模型
    torch.save(model.state_dict(), "../../model_saver/model_method2.pth")  # 保存模型参数，这种方式模型加载时，需要重新实例化模型，再加载参数
