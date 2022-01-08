'''
@author：fc
@date：  2022/1/8
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# transform的使用：将特定格式的图片或是numpy数组转换成tensor
# 主要使用的方法有：ToTensor,reSize
"""
from torchvision import transforms
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
import numpy as np
if __name__ == '__main__':
    wirter = SummaryWriter("../../log/tensorboard/")
    # I、writer.add_image()中使用张量，张量的得到需要使用transforms.ToTensor()工具箱,不需要指定数据格式，因为张量格式本生是C、H、W,这个是本代码文件的主要学习内容
    img = Image.open("G:/projects/PycharmProjects/Dataset/hymenoptera_data/train/bees/196430254_46bd129ae7.jpg")
    tensor_trans = transforms.ToTensor()
    img_tensor = tensor_trans(img)
    wirter.add_image("test_cv1", img_tensor, 1)
    wirter.close()

    # II、writer.add_image()中使用numpy数组，这里numpy数组的得到是通过cv2,cv2.imread(path)直接就可以了，需要指定数据格式，numpy读出来格式是H、W、C
    img_array1 = cv2.imread("G:/projects/PycharmProjects/Dataset/hymenoptera_data/train/bees/196430254_46bd129ae7.jpg")
    wirter.add_image("test_cv2",img_array1,2,dataformats="HWC")
    wirter.close()

    # III、writer.add_image()中使用numpy数组,这里numpy数组的得到是通过numpy转换PIL.Image.open(path)得到，需要指定数据格式，numpy读出来格式是H、W、C
    img = Image.open("G:/projects/PycharmProjects/Dataset/hymenoptera_data/train/bees/196430254_46bd129ae7.jpg")
    img_array2 = np.array(img)
    print(f"图片形状：{img_array2.shape}")
    wirter.add_image("test_cv3", img_array2, 3, dataformats="HWC")  # 参数变量需要为张量、numpy数组,1代表的时步骤
    wirter.close()
