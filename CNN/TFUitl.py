"""
@author：fc
@date：  2021/12/9
@contact：675435108@qq.com
"""
"""
文件内容&功能简要：工具类
tensorflow使用GPU加速时，显卡的选择和内存的按需分配、以及日志输出的约束
"""

import warnings
warnings.filterwarnings("ignore") # 这个起作用要放在tensorflow导入之前,直接使用tensorflow限制输出感觉无效
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 这个起作用要放在tensorflow导入之前，直接使用tensorflow选择感觉无效
import tensorflow as tf
print(tf.test.is_gpu_available())
class TFTool:
    """
        gpu_id：显卡号
        memory_size:最大使用内存，单位G
    """
    def __init__(self, gpu_id, memory_size=None):
        self.gpu_id = gpu_id  # 选择是应用的显卡号
        self.memory_size = memory_size  # 设置使用显卡最大内存，单位为G
        self.memory_allocation()

    def memory_allocation(self):
        # 获取主机运算设备类型：这里获取的是GPU的，也可以将device_type=值该为CPU
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        print(f"可用加速卡有：{gpus}")
        # 设置使用的显卡，这里四块都设置可用
        # tf.config.experimental.set_visible_devices(devices=gpus[0:],device_type='GPU')
        # 只使用其中一块
        tf.config.experimental.set_visible_devices(devices=gpus[self.gpu_id], device_type='GPU')
        # tensorflow使用gpu加速时内存自动分配
        # tf.config.experimental.set_memory_growth(gpus[self.gpu_id], True)
        if gpus:
            try:
                for i in range(len(gpus)):
                    tf.config.experimental.set_memory_growth(gpus[i], True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print("真实GPU个数：",len(gpus), "逻辑gpu个数：", len(logical_gpus))
            except RuntimeError as e:
                print(e)
        if self.memory_size is not None:
            # 限制使用最大内存
            tf.config.experimental.set_virtual_device_configuration(
                gpus[self.gpu_id],  # 该显卡使用的最大内存限制
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * self.memory_size)]
            )


if __name__ == '__main__':
    tfTool1 = TFTool(0)
