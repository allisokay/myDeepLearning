'''
@author：fc
@date：  2022/1/7
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# pytorch中数据加载的抽象类Dataset的继承
"""
from torch.utils.data import Dataset
from PIL import Image
import os
import random
class MyDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.label_path = os.path.join(root_dir,label_dir)
        self.imgs_path = os.listdir(self.label_path)

    def __getitem__(self, idx):
        """
        返回图片和图片对应的标签
        :param idx: 下标
        :return:
        """
        img_name = self.imgs_path[idx]
        img_path = os.path.join(self.label_path,img_name)
        img = Image.open(img_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.imgs_path)


if __name__ == '__main__':
    idx = random.randint(1, 124)
    root_dir = "G:/projects/PycharmProjects/Dataset/hymenoptera_data/train/"
    # 蚂蚁数据集
    ant_label_name = "ants"
    ant_data = MyDataset(root_dir,ant_label_name)
    print(f"the length of ant dataset is {len(ant_data)}")
    sigle_data = ant_data[idx]  # 调用MyDataset中的getItem方法，返回单个数据和标签
    sigle_data[0].show()
    print(sigle_data[1])
    # 蜜蜂数据集
    bees_label_name = "bees"
    bees_data = MyDataset(root_dir,bees_label_name)
    print(f"the length of bees dataset is {len(bees_data)}")
    bees_data[idx][0].show()
    print(bees_data[idx][1])
    #  让我诧异的是两个数据集可以拼接
    data= ant_data + bees_data
    print(f"拼接后的数据集长度为：{len(data)}")
    data[idx][0].show()
    print(data[idx][1])