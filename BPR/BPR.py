"""
@author：67543
@date：  2022/2/21
@contact：675435108@qq.com
"""
import heapq
import multiprocessing
import time
import numpy as np
import torch

import metrics
data_generator = 0


def test_one_user(x):
    rating = x[0]
    u =x[1]
    ITEM_NUM = data_generator.n_items
    Ks = [20,40,60,80,100]
    try:
        train_itmes = data_generator.train_items[u] # 训练集中user交互过的item
    except  Exception:
        train_itmes = []
    user_pos_test = data_generator.test_set[u]  # 测试集中用户真实交互后的item
    all_items = set(range(ITEM_NUM))
    # 所有的items-训练集中用户交互过的item==>测试集中用户交互过item和未交互过的item被留下
    # 测试集中交互过的item用户做验证
    test_items = list(all_items-set(train_itmes))

    def ranklist_by_sorted(user_pos_test,test_items,rating,Ks):
        """
         args:
            user_pos_test:测试集中用户真实交互后的item,但训练集中该item是没有出现的
            test_items:待选的item
            rating:用户的所有预测评分
            Ks:Top-K
        """
        item_score = {} # 存储用户对未访问过的item的评分
        for i in test_items:
            item_score[i] = rating[i]
        # 直接返回topK 最大的命中情况，则当K是小值是，命中情况也自然返回了
        # 比如这里直接返回50个评分从大到小的值，评分大的值可能是没有访问过的（没命中）
        K_max = max(Ks)
        # heapq.nlargest()：在item_score中找到K_max个最大的元素,这里返回的应该是字典
        # 找到K_max个tem_score的value最大的元素
        K_max_item_score = heapq.nlargest(K_max,item_score,key=item_score.get)
        r = []
        for i in K_max_item_score:
            # 如果i出现在测试集中用户真实交互后的item集合中，命中
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = 0
        return r,auc

    def get_performance(user_pos_test,r,auc,Ks):
        """
        aim:计算模型性能指标
        args:
            user_pos_test:测试集中用户真实交互后的item
            r:            r = [1,0,1] 表示预测TOP-K是否命中
            auc:          auc =0 标量
            ks:           TOP-K
        """
        # 准确率、召回率，命中率，Map值计算
        precision,recall,ndcg,hit_ration,map = [],[],[],[],[]
        for K in Ks:
            precision.append(metrics.precision_at_k(r,K))
            recall.append(metrics.recall_at_k(r,K,len(user_pos_test)))
            ndcg.append(metrics.ndcg_at_k(r,K))
            hit_ration.append(metrics.hit_at_k(r,K))
            map.append(metrics.map_at_k(r,K,len(user_pos_test)))
        return {
            'precision':np.array(precision),
            'recall':np.array(recall),
            'ndcg':np.array(ndcg),
            'hit_ratio':np.array(hit_ration),
            'MAP':np.array(map),
            'auc': auc
        }
    r,auc = ranklist_by_sorted(user_pos_test,test_items,rating,Ks)
    return get_performance(user_pos_test,r,auc,Ks)


class BPR(torch.nn.Module):
    def __init__(self,n_user,n_item,embeding_size=4,l2_reg_embedding=0.00001,device='cpu'):
        super(BPR, self).__init__()
        self.n_user = n_user
        self.n_items = n_item
        self.embedding_size = embeding_size
        self.device = device
        self.l2_reg_embedding = l2_reg_embedding
        # nn.ModuleDict:将常用操作封装到模块字典中，通过下标调用操作
        # 符合BPR概念，每个用户都会是一个Embedding
        self.embedding_dict = torch.nn.ModuleDict({
            'user_emb':self.create_embedding_matrix(n_user,embeding_size),
            'item_emb':self.create_embedding_matrix(n_item,embeding_size)
        })
        self.to(device)  # 这句话有些鸡肋

    def forward(self,input_dict):
        users,pos_items,neg_items = input_dict['users'],input_dict['pos_items'],input_dict['neg_items']
        user_vector = self.embedding_dict['user_emb'](users)
        # pos_items_vector和 neg_items_vector对应的Embedding权重参数矩阵是一样的
        pos_items_vector = self.embedding_dict['item_emb'](pos_items)
        neg_items_vector = self.embedding_dict['item_emb'](neg_items)
        # matmul是矩阵数数相乘,mul也是位位相乘,
        # 最重要的也就是这一步了user_vector和item_vector位位相乘，再对每一行的的数相加返回结果
        # 算是提取用户和item特征吧
        rui = torch.sum(torch.mul(user_vector,pos_items_vector),dim=-1,keepdim=True)
        ruj = torch.sum(torch.mul(user_vector,neg_items_vector),dim=-1,keepdim=True)
        # 求2范数（L2正则:矩阵和其转置矩阵乘积的最大特征值得平方根）,矩阵的二范数（矩阵的特征值可以用torch.eig(input,eigenvectors=True)）求
        # 正则化使得学习的模型参数值较小，是常用防止过拟合的常用手段
        emb_loss = torch.norm(user_vector)**2+torch.norm(pos_items_vector)**2+torch.norm(neg_items_vector)**2
        return rui,ruj,emb_loss

    def fit(self,learning_rate = 0.001,batch_size=500,epochs=15,verbose=5):
        """
         模型训练，定义到了模型中
        """
        self.data_generator = data_generator
        model=self.train()  # 自实例化，类似于model = BPR()
        # 一般说来，这里应该定义为激活函数，而且应该在模型中（写在forward中）
        loss_func = torch.nn.LogSigmoid()
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.0)
        """
         训练前设置，得到每个epoch中的batch数=训练总的交互数/batch_size
        """
        sample_num = data_generator.n_train     # 训练文本交互数
        n_batch = (sample_num-1) //batch_size+1  # //表示除数,得到每个epoch中的batch数
        print(f"训练:共{ sample_num }个交互数,{epochs}个epoch,每个step的batch_size={batch_size}，每个epoch有{ n_batch }个step")
        """
         开始训练
        """
        for epoch in range(epochs):
            # 总的损失 =  矩阵分解损失+Embedding损失
            total_loss ,total_mf_loss,total_emb_loss = 0.0,0.0,0.0
            with torch.autograd.set_detect_anomaly(True):
                # 存储每个epoch的时间
                start_time = time.time()
                for index in range(n_batch):
                    # 为users中的每一个用户采样一个正样本和一个负样本存储在pos_items和neg_items中
                    # 正样本即用户交互过的item，负样本即用户没有交互过的item
                    users,pos_items,neg_items = data_generator.sample(batch_size)
                    # gpu向量化
                    users = torch.from_numpy(np.array(users)).to(self.device).long()
                    pos_items = torch.from_numpy(np.array(pos_items)).to(self.device).long()
                    neg_items = torch.from_numpy(np.array(neg_items)).to(self.device).long()
                    input_dict =  {'users':users,'pos_items':pos_items,'neg_items':neg_items}
                    rui,ruj,emb_loss = model(input_dict)
                    optimizer.zero_grad()
                    mf_loss = -loss_func(rui-ruj).mean()
                    reg_emb_loss = self.l2_reg_embedding*emb_loss/batch_size
                    loss = mf_loss+reg_emb_loss
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    total_mf_loss = total_mf_loss+mf_loss.item()
                    total_emb_loss = total_emb_loss + reg_emb_loss.item()
                    total_loss = total_loss+loss.item()
                 # 结果输出
                if verbose>0:
                    epoch_time = time.time()-start_time
                    # 每个step的总损失、分解损失，Embedding损失
                    print(f'epoch {epoch}：{epoch_time:.2f}s,训练总损失({total_loss/n_batch:.4f})=矩阵分解损失({total_mf_loss/n_batch:.4f})+嵌入损失({total_emb_loss/n_batch:.4f})')

            # epoch是轮数的整倍数时，做一个评价指标输出
            if verbose>0 and epoch%verbose ==0:
                start_time = time.time()
                result = self.test(batch_size)
                eval_time = time.time()-start_time
                print("epoch %d %.2fs 测试precision=[%.6f %.6f], recall="
                  "[%.6f %.6f] ,ndcg=[%.6f %.6f],hit_ration=[%.6f %.6f]，"
                  "MAP=[%.6f %.6f],auc=%.6f"%
                  (epoch,eval_time,
                   result['precision'][0], result['precision'][-1],
                   result['recall'][0], result['recall'][-1],
                   result['ndcg'][0], result['ndcg'][-1],
                   result['hit_ratio'][0], result['hit_ratio'][-1],
                   result['MAP'][0], result['MAP'][-1],
                   result['auc']
                   ))
        print("")

    def test(self,batch_size=256):
        model = self.eval()
        cores = multiprocessing.cpu_count() // 2
        pool  = multiprocessing.Pool(cores)
        Ks = [20,40,60,80,100]
        ITEM_NUM = data_generator.n_items
        result = {
            'precision':np.zeros(len(Ks)),
            'recall':np.zeros(len(Ks)),
            'ndcg':np.zeros(len(Ks)),
            'hit_ratio': np.zeros(len(Ks)),
            'MAP': np.zeros(len(Ks)),
            'auc': 0.
        }
        u_batch_size = batch_size
        test_users = list(data_generator.test_set.keys())
        n_test_users = len(test_users)
        # 用户批数,测试文本中共有n_test_users=52639个用户//一批用户有batch_size=2000个用户=27批用户
        n_user_batchs = (n_test_users-1) // u_batch_size+1
        count = 0
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id*u_batch_size
            end = (u_batch_id+1)*u_batch_size
            # 取出用户，第一次取0~1999索引对应的用户，第二从20000~3099，一直完
            # 每个用户都可以取到
            user_batch = test_users[start:end]
            all_item = range(ITEM_NUM)
            user_batch = torch.from_numpy(np.array(user_batch)).to(self.device).long()
            all_item = torch.from_numpy(np.array(all_item)).to(self.device).long()
            rate_batch = model.rating(user_batch,all_item).detach().cpu()
            # 将两个迭代器捆绑成list，list有两千个值
            # 每个值为一个元组（[xx,xxx,...,xxx],user）,元组第一个值是该user对测试集中每个item的预测评分
            user_batch_rating_uid = zip(rate_batch.numpy(),user_batch.detach().cpu().numpy())
            batch_result =  pool.map(test_one_user,user_batch_rating_uid)
            count += len(batch_result)
            for re in batch_result:
                result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['hit_ratio'] += re['hit_ratio'] / n_test_users
                result['MAP'] += re['MAP'] / n_test_users
        assert  count == n_test_users
        pool.close()
        return result


    def rating(self,user_batch,all_item):
        """
          矩阵乘法后每个用户都和测试集中的item建立了关联得到2000*num(itme)数的矩阵
          也就是测试集中每个用户对对所有item的评分
        """
        user_vector = self.embedding_dict['user_emb'](user_batch)
        pos_item_vector = self.embedding_dict['item_emb'](all_item)
        return torch.mm(user_vector,pos_item_vector.t())

    def create_embedding_matrix(self,vocabulary_size,embedding_size,init_std=0.0001,sparse=False):
        """
            字面意思是：创建嵌入矩阵
        """
        # 设置embedding的词袋数，关联特征数
        embedding = torch.nn.Embedding(vocabulary_size,embedding_size,sparse=sparse)
        # 设置Embedding的权重矩阵参数服从均值维0，方差为0.0001的正态分布
        torch.nn.init.normal_(embedding.weight,mean=0,std=init_std)
        return embedding
