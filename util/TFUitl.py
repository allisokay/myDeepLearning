"""
@author：fc
@date：  2021/12/9
@contact：675435108@qq.com
"""
"""
文件内容&功能简要：工具类
tensorflow使用GPU加速时，显卡的选择和内存的按需分配、以及日志输出的约束
"""
print("---"*12+"TensorFlow-gpu信息输出开始"+"---"*12)
import warnings
warnings.filterwarnings("ignore")  # 这个起作用要放在tensorflow导入之前,直接使用tensorflow限制输出感觉无效
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 这个起作用要放在tensorflow导入之前，直接使用tensorflow选择感觉无效
import tensorflow as tf
print(f"gpu加速是否可用：{tf.test.is_gpu_available()}")
# 获取主机运算设备类型：这里获取的是GPU的，也可以将device_type=值该为CPU
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(f"可用加速卡有：{gpus}")
# tensorflow使用gpu加速时内存自动分配
if gpus:
   try:
      for i in range(len(gpus)):
          tf.config.experimental.set_memory_growth(gpus[i], True)
          logical_gpus = tf.config.experimental.list_logical_devices('GPU')
          print("真实GPU个数：", len(gpus), "逻辑gpu个数：", len(logical_gpus))
   except RuntimeError as e:
          print(e)
print("---"*24+"TensorFlow-gpu信息输出结束\n")
