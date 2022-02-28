"""
@author：67543
@date：  2022/2/18
@contact：675435108@qq.com
"""
import warnings
warnings.filterwarnings("ignore")
import torch

from load_data import Data
import BPR as bpr
from BPR import BPR

if __name__ == '__main__':
    """
    1.加载数据,
    """
    file_path = "../public_dataset/amazon-book"
    data_generator = Data(file_path)
    bpr.data_generator = data_generator
    """
      2.获取模型
    """
    # 2.1训练设备选择
    use_gpu = False
    if use_gpu and torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = 'cpu'
    # 2.2 加载模型
    model = BPR(data_generator.n_users,data_generator.n_items,embeding_size=10,l2_reg_embedding=0.00001,device=device)
    """
    3.模型训练和测试
    """
    model.fit(learning_rate=0.001,batch_size=2000,epochs=50,verbose=5)