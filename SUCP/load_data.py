# -*- coding: utf-8
"""
@author：67543
@date：  2022/3/10
@contact：675435108@qq.com
"""
from collections import defaultdict
from scipy import sparse
import numpy as np
import os

import time


class Data(object):
    def __init__(self,dataset):
        data_dir = "../public_dataset/" + dataset + "/"
        self.dataset = dataset
        self.data_dir = data_dir
        self.size_file = os.path.join(data_dir, dataset + "_data_size.txt")
        self.checkins_file = os.path.join(data_dir, dataset + "_checkins.txt")
        self.train_file = os.path.join(data_dir, dataset + "_train.txt")
        self.turn_file = os.path.join(data_dir, dataset + "_tune.txt")
        self.test_file = os.path.join(data_dir, dataset + "_test.txt")
        self.poi_file = os.path.join(data_dir, dataset + "_poi_coos.txt")
        self.social_file = os.path.join(data_dir, dataset + "_social_relations_70.txt")
        with open(self.size_file, 'r') as f:
            user_nums, poi_nums = f.readlines()[0].strip("\n").split()
            self.user_nums, self.poi_nums = int(user_nums), int(poi_nums)
        self.train_tuples = set()

    def get_train_data(self):
        """
        加载训练文本，返回
            稀疏矩阵字典， 训练元组
            训练矩阵：隐式矩阵（表示用户和poi存在交互)，显式表示存在交互次数
           """
        # 创建一个user_nums×poi_nums的稀疏矩阵，dok_matrix返回的是字典，key是矩阵
        sparse_train_matrix = sparse.dok_matrix((self.user_nums, self.poi_nums))
        train_matrix_im, train_matrix_ex = np.zeros((self.user_nums, self.poi_nums)), np.zeros((self.user_nums, self.poi_nums))
        train_tuples = set()
        with open(self.data_dir + self.dataset + "_train.txt") as f:
            for line in f.readlines():
                uid, lid, freq = line.strip().split()
                uid, lid, freq = int(uid), int(lid), int(freq)
                sparse_train_matrix[uid, lid] = freq
                train_matrix_ex[uid, lid] = freq
                train_matrix_im[uid, lid] = 1.0
                train_tuples.add((uid, lid))
        self.train_tuples = train_tuples
        print("    训练文本加载完成")
        with open("./process/sparse_train_matrix.txt",'w') as f:
            f.write(str(sparse_train_matrix))
        with open("./process/train_tuples.txt",'w') as f:
            f.write(str(train_tuples))
        with open("./process/train_matrix.txt",'w') as f:
            f.write(str(train_matrix_im))
        with open("./process/train_matrix2.txt",'w') as f:
            f.write(str(train_matrix_ex))
        return sparse_train_matrix,train_tuples,train_matrix_im,train_matrix_ex

    def get_checkin_data(self):
        """
         加载签到数据集,返回
         时间段字典（key=(hour,uid,lid)）组成的list, value=访问次数
         休息日，工作日字典（key=(hour,uid,lid),value = 访问次数
        """
        train_tuples_with_day = defaultdict(int)  # key整形字典
        train_tuples_with_time = defaultdict(int)
        with open(self.data_dir + self.dataset + "_checkins.txt", 'r') as f:
            for line in f.readlines():
                uid, lid, checkin_time = line.strip().split()
                uid, lid, checkin_time = int(uid), int(lid), float(checkin_time)
                # hour的大小可以得出工作时间和闲暇时间，那双休是哪天呢？5,6？
                if (uid, lid) in self.train_tuples:
                    hour = time.gmtime(checkin_time).tm_hour  # 细小的差别会要我得命
                    train_tuples_with_time[(hour, uid, lid)] += 1.0
                    if 8 <= hour < 18:  # 工作时间
                        hour = 0
                    elif hour >= 18 or hour < 8:  # 休息时间
                        hour = 1
                    train_tuples_with_day[(hour, uid, lid)] += 1.0
        # 为0-23hour都创建一个user_nums*poi_nums稀疏矩阵矩阵
        sparse_train_matrices = [sparse.dok_matrix((self.user_nums, self.poi_nums)) for _ in range(24)]
        for (hour, uid, lid), freq in train_tuples_with_time.items():
            # (x/(1+x)) 0~∞逐渐趋近1，得到用户该时对该POI的访问概率？
            sparse_train_matrices[hour][uid, lid] = 1.0 / (1.0 + 1.0 / freq)
        # 为工作日和周末创建user_nums*poi_nums稀疏矩阵矩阵
        sparse_train_matrix_WT = sparse.dok_matrix((self.user_nums, self.poi_nums))
        sparse_train_matrix_LT = sparse.dok_matrix((self.user_nums, self.poi_nums))
        for (hour, uid, lid), freq in train_tuples_with_day.items(): # 细小的差别会要我的命，这里我写的是preq,下面freg
            if hour == 0:
                sparse_train_matrix_WT[uid, lid] = freq
            elif hour == 1:
                sparse_train_matrix_LT[uid, lid] = freq
        print("    签到文本加载完成")
        with open("./process//sparse_train_matrices.txt", 'w') as f:
            f.write(str(sparse_train_matrices))
        with open("./process//sparse_train_matricx_WT.txt", 'w') as f:
            f.write(str(sparse_train_matrix_WT))
        with open("./process//sparse_train_matricx_LT.txt", 'w') as f:
            f.write(str(sparse_train_matrix_LT))
        return sparse_train_matrices,sparse_train_matrix_WT,sparse_train_matrix_LT
     
    def get_test_data(self):
        ground_truth = defaultdict(set) # value为()
        with open(self.test_file,'r')as f:
            for line in f.readlines():
                uid,lid,_ = line.strip().split()
                uid,lid = int(uid),int(lid)
                ground_truth[uid].add(lid)
        print("    测试文本加载完成，")
        with open("./process//ground_truth.txt", 'w') as f:
            f.write(str(ground_truth))
        return ground_truth

    def get_poi_coos_data(self):
        poi_coos = dict()
        with open(self.poi_file,'r') as f:
            for line in f.readlines():
                lid,lat,lng = line.strip().split()
                lid,lat,lng = int(lid),float(lat),float(lng)
                poi_coos[lid] = (lat,lng)
        print("    位置（poi）文本加载完成")
        with open("./process//poi_coos.txt", 'w') as f:
            f.write(str(poi_coos))
        return poi_coos

    def get_social_data(self):
        social_matrix = np.zeros((self.user_nums,self.poi_nums))
        with open(self.social_file,'r') as f:
            for line in f.readlines():
                uid1,uid2,_,_,_,_,_= line.strip().split()
                uid1,uid2 = int(uid1),int(uid2)
                social_matrix[uid1,uid2] = 1.0
                social_matrix[uid2,uid1] = 1.0
        print("    用户社交关系文本加载完成")
        with open("./process//social_matrix.txt", 'w') as f:
            f.write(str(social_matrix))
        return social_matrix

