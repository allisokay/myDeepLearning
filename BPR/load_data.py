"""
@author：67543
@date：  2022/2/18
@contact：675435108@qq.com
"""
import random
import numpy as np

class Data(object):
    def __init__(self, file_path, batch_size=256):
        self.file_path = file_path
        self.batch_size = batch_size
        # 用户总数，物品总数
        self.n_users, self.n_items = 0, 0
        # 训练文本中的交互总数，测试文本中的交互总数
        self.n_train, self.n_test = 0, 0
        # 训练文本以及测试文本中每个用户和其访问序列组成的字典
        self.train_items, self.test_set = {}, {}
        # 负采样池？，训练集中存在的用户
        self.neg_pools, self.exist_users = {}, []
        # 训练数据构建
        self.get_train_data(self.file_path + '/train.txt')
        # 测试数据构建
        self.get_test_data(self.file_path + '/test.txt')
        # 输出统计
        self.print_statistics()

    def get_train_data(self, train_data_path):
        """
        train_data_path:训练文本路径
        得到训练文本中的
            用户数（n_users：最大用户Id）、物品数（n_items:最大ItemId）、
            访问总次数（n_train:每个用户的访问次数之和）、
            存在的用户列表:exits_users(训练文本中不一定包含所有用户)、
            每个用户和其访问序列组成的字典:train_items
        """
        with open(train_data_path, 'r') as f:
            for line in f.readlines():  # 逐行读取
                if len(line) > 0:
                    l = line.strip('\n').split(' ')  # 去掉换行符后以空格切割
                    userId = int(l[0])
                    items = [int(i) for i in l[1:]]
                    self.n_users = max(self.n_users, userId)
                    self.n_items = max(self.n_items, max(items))
                    self.n_train += len(items)
                    self.exist_users.append(userId)
                    self.train_items[userId] = items

    def get_test_data(self, test_data_path):
        """
        test_data_path:测试文本路径
        目的：得到测试文本中的
            访问总次数：n_test(每个用户的访问次数之和)、
            每个用户访问序列组成的字典：test_set,
            更新物品总数：n_items
            ？：奇异的是为什么不更新用户总数，测试文本中也存在userId呀
        """
        with open(test_data_path, 'r') as f:
            for line in f.readlines():
                if len(line) > 0:
                    try:
                        l = line.strip('\n').split(' ')
                        userId = int(l[0])
                        items = [int(i) for i in l[1:]]
                    except Exception as e:
                        # print("训练数据处理完毕")  # 存在错误：invalid literal for int() with base 10: ''
                        continue
                    self.n_test += len(items)
                    self.test_set[userId] = items
                    self.n_items = max(self.n_items,len(items))

    def print_statistics(self):
        # user、item都是从0开始编号，所以+1
        self.n_items += 1
        self.n_users += 1
        print(f"用户数={self.n_users},item数={self.n_items}")
        print(f"共{self.n_train+self.n_test}次交互=训练集交互数({self.n_train})+测试集交互数({self.n_test}),sparsity={(self.n_train+self.n_test)/(self.n_users*self.n_items):.5f}")

    def sample(self,batch_size=256 ):
        """
        返回 UserId,PositiveItemId negativeItemId
        """
        if batch_size <= self.n_users:
            # 从exist_users中随机采2000个唯一数
            users = random.sample(self.exist_users,batch_size)
        else:
            users = [random.choice(self.exist_users) for _ in range(batch_size)]

        def sample_pos_item_for_u(u,num):
            """
             函数中的子函数，可在函数中调用，不能跳出函数调用，函数外调用要先将子函数返回
             目的:为用户u采正向item,采满num个。这里的正向item即用户交互过的item
            """
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                # 随机生成下标值（范围在0-该用户的交互列表的长度），这样取出的值一定是正的，即交互过的
                pos_id = np.random.randint(low=0,high=n_pos_items,size=1)[0]
                pos_item = pos_items[pos_id]
                if pos_item not in pos_batch:
                    pos_batch.append(pos_item)
            return pos_batch

        def sample_neg_item_for_u(u,num):
            """
                为用户用采num个负向item，这里的负向item是用户没交互过的
            """
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                # 这里得到的neg_item直接就是真实训练文本中用户未访问过的item了，有与用户交互过的，也有没交互过的
                neg_item = np.random.randint(low=0,high=self.n_items,size=1)[0]
                # 到这里，我知道init中n_item(item总数）也需要在测试文本中更新的原因了
                if neg_item not in self.train_items[u] and neg_item not in neg_items:
                    neg_items.append(neg_item)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_item_for_u(u, 1)
            neg_items += sample_neg_item_for_u(u,1)
        return users,pos_items,neg_items