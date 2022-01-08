'''
@author：fc
@date：  2022/1/8
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# tensorboard使用尝试，pytorch1.1中也引入了tensorboard
"""
import warnings
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

warnings.filterwarnings("ignore")


def use_add_scaler():
    writer = SummaryWriter("../../log/tensorboard/")
    for i in range(100):
        writer.add_scalar("y=6x^3+4x^2+8", 6 * pow(i, 3) + 4 * pow(i, 2) + 8, i)
    writer.close()

def use_add_image():
    writer = SummaryWriter("../../log/tensorboard/")
    img = Image.open("G:/projects/PycharmProjects/Dataset/hymenoptera_data/train/bees/196430254_46bd129ae7.jpg")
    img_array = np.array(img)
    print(f"图片形状：{img_array.shape}")
    writer.add_image("train(add_image)",img_array,2,dataformats="HWC")  # 参数变量需要为张量、numpy数组,1代表的时步骤
    writer.close()


if __name__ == '__main__':
    use_add_scaler()
    use_add_image()