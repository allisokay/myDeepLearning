"""
@author：67543
@date：  2022/2/23
@contact：675435108@qq.com
"""
import numpy as np


def precision_at_k(r,k):
    """
    计算精确率
    args:
        r:r = [1,0,1,...] 表示预测TOP-K是否命中,1代表预测的item出现在测试集用户交互过
        的items列表中
        k: 取前k个
    """
    assert k >= 1
    # 这里的精确率等于推荐的k个item中，出现在测试集用户交互过的items列表中的个数/k

    r = np.asfarray(r)[:k]
    return  np.mean(r)


def recall_at_k(r,k,all_pos_num):
    """
       计算召回率
       args:
           r:r = [1,0,1,...] 表示预测TOP-K是否命中,1代表预测的item出现在测试集用户交互过
           的items列表中
           k: 取前k个
           all_pos_num:用户测试集中该用户交互过的item总数
    """
    r = np.asfarray(r)[:k]
    # 这里的召回率等于推荐的k个item，出现在测试集用户交互过的items列表中的个数 /
    # 测试集中用户交互过的items总数
    # 和精确率相比，被除数不一样，除数都是前k个推荐的命中个数
    return np.sum(r)/all_pos_num


def ndcg_at_k(r,k,method=1):
    """
    计算 ndcg
       args:
           r:r = [1,0,1,...] 表示预测TOP-K是否命中,1代表预测的item出现在测试集用户交互过
           的items列表中
           k: 取前k个
    """
    # 计算ndcg=dcg/idcg(理论上的dcg，即命中的要排在首位,2位即dcg_max)，计算dcg有两种方法，第一种（method）是通用类型
    # 第二种是针对打分只有两档的情况，换句话来说就是命中和没命中
    # 但method2的公式中，我是不理解2^r-1为什么等于r的===>发现当r中的值都是1,0时，r=2^r-1
    def dcg_at_k(r, k, method=1):
        """
         计算dcg
        """
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                # numpy中支持长度和维度相同的list直接相除
                return r[0] + np.sum(r[1:]) / np.log2(np.arange(2, r.size + 1))
            elif method == 1:
                # 其实我是不太理解公式2中的(2^r-1)等于r的
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method的值必须是0或者1')
        return 0
    dcg_max = dcg_at_k(sorted(r,reverse=True),k,method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r,k,method)/dcg_max


def map_at_k(r,k,all_pos_num):
    """
     计算：mean average percision,但要先计算average_percision:ap_at_k
     map=ap/测试样本中用户u所有交互过的item长度
    """
    def ap_at_k(r,k):
        """
         average_percision=每k(1,2,3)的命中个数/k的累加
        """
        hits = 0
        sum_pre = 0
        for i in range(k):
            if r[i] == 1:
                hits += 1
                sum_pre += hits / (i+1.0)
        return sum_pre

    return ap_at_k(r,k) / all_pos_num



def hit_at_k(r,k):
    """
     这里的命中是推荐的k个item中，只要中一个就中，那还不如用精确率
    """
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.





