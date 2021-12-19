'''
@author：fc
@date：  2021/12/9
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# CNN基础模型（卷积+池化+卷积+池化+全连接+全连接+高斯连接）的实现,模型图片在CNN目录下，基于tensorflow1.14
"""
import warnings
warnings.filterwarnings("ignore")  # 这个起作用要放在tensorflow导入之前,直接使用tensorflow限制输出感觉无效
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 这个起作用要放在tensorflow导入之前，直接使用tensorflow选择感觉无效
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class CNN:
    """
     矩阵的单个样本变形后的长宽、通道数和单个样本总长度，标签类别数
    """

    def __init__(self, width, height, channels, sigle_sample_total_length, labels_length):
        self.input_width = width
        self.input_height = height
        self.input_channels = channels
        self.sigle_sample_total_length = sigle_sample_total_length
        self.labels_length = labels_length

    def stractures(self, input_x=None):
        """
           1.输入层
        """
        input_x = input_x
        """
            2.隐藏层（内包含多层网络）
             我认为模型中：学习率、卷积核大小、优化器、步长这些值是固定的，不该再改动
        """
        """
           第一个层：卷积+激活+池化
        """
        # 2.1.1 定义卷积核并初始化值满足均值为0，方差为0.1的正态分布，并将其加入TensorFlow中需要梯度优化的变量中
        layer1_conv_kernel = tf.Variable(tf.truncated_normal([5, 5, self.input_channels, 32], stddev=0.1))
        # 2.1.2 定义偏置并初始化其值满足0，0.1的正太分布，以便激活后易于分类,加入TensorFlow需要梯度优化变量中
        layer1_bias = tf.Variable(tf.truncated_normal([32], stddev=0.1))
        # 2.1.3 设置卷积核移动的步长和存在剩余单元格时是否填充数据矩阵input_x：是
        layer1_strides, layer1_padding = [1, 1, 1, 1], 'SAME'
        # 2.1.4 卷积运算+偏置
        layer1_conv = tf.nn.conv2d(input_x, layer1_conv_kernel, strides=layer1_strides,
                                   padding=layer1_padding) + layer1_bias
        # # 2.1.5 卷积结果激活
        layer1_conv_res1_activ = tf.nn.relu(layer1_conv)
        # 2.1.6 激活后最大池化
        # 池化前先设置池化窗口大小为2*2,移动步长为2*2,填充卷积后的矩阵
        layer1_kSize, layer1_max_pool_strides = [1, 2, 2, 1], [1, 2, 2, 1]
        # 2.1.7 开始池化,池化后还可以激活，这里先不加
        layer1_max_pool = tf.nn.max_pool(layer1_conv_res1_activ, ksize=layer1_kSize, strides=layer1_max_pool_strides,
                                         padding=layer1_padding)
        layer1_th_network = layer1_max_pool  # 池化不用激活
        """
            第二层:卷积+激活层+池化
        """
        # 卷积核定义，经过第一层后，通道数变成来32
        layer2_conv_kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        # 定义骗置
        layer2_bias = tf.Variable(tf.truncated_normal([64], stddev=0.1))
        # 卷积运算并激活
        layer2_conv_activ = tf.nn.relu(
            tf.nn.conv2d(layer1_th_network, layer2_conv_kernel, strides=[1, 1, 1, 1], padding='SAME')) + layer2_bias
        # 池化，池化后还可以激活，这里先不加
        layer2_max_pool = tf.nn.max_pool(layer2_conv_activ, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        layer2_th_network = layer2_max_pool  # 池化不用激活
        """
           第三层：全连接层1
        """
        # 全连接参数的数量是可以任意的，这里可以调节，
        layer3_weight = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
        layer3_bias = tf.Variable(tf.truncated_normal([1024]))
        # 矩阵乘法
        layer2_th_network_flat = tf.reshape(layer2_th_network, [-1, 7 * 7 * 64])
        layer3_fc = tf.matmul(layer2_th_network_flat, layer3_weight) + layer3_bias
        layer3_th_network = tf.nn.relu(layer3_fc)
        """
           第四层：全连接层2
        """
        # 全连接参数的数量是可以任意的，
        layer4_weight = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.1))
        layer4_bias = tf.Variable(tf.truncated_normal([512]))
        layer4_fc = tf.matmul(layer3_th_network, layer4_weight) + layer4_bias
        layer4_th_network = tf.nn.relu(layer4_fc)
        """
           第五层：高斯连接层
        """
        layer5_weight = tf.truncated_normal([512, 10], stddev=0.1)
        layer5_bias = tf.truncated_normal([10])
        layer5_fc = tf.matmul(layer4_th_network, layer5_weight) + layer5_bias
        # leNet5时softmax，还可以用其他的分类函数（如，softmax_cross_entorpy_with_logits）
        # layer5_th_network = tf.nn.softmax_cross_entropy_with_logits(layer5_fc)
        # 先softmax再计算交叉熵的方法
        layer5_th_network = tf.nn.softmax(layer5_fc)
        pre = layer5_th_network
        return pre

    def train(self, input_x=None, input_y=None):
        # 为输入的矩阵和标签生成占位符,None表示需要的样本数不确定
        x = tf.placeholder(tf.float32, [None, self.sigle_sample_total_length])
        # 这里变形是有些奇怪的，因为全连接时还是要变回来
        train_x = tf.reshape(x, [-1, self.input_width, self.input_height, self.input_channels]) # 这里的变形值不能再为x,所以我将x改成来train_x,否则就是输入维度错误
        y = tf.placeholder(tf.float32, [None, self.labels_length])
        pre = self.stractures(train_x)
        # 损失函数
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=pre)
        # 优化器,学习率0.001
        opt = tf.train.AdamOptimizer(1e-4).minimize(loss)
        tf.summary.histogram(name="loss", values=loss)

        # 准确率
        corr_pre = tf.equal(tf.argmax(input_y, 1), tf.argmax(pre, 1))
        acc = tf.reduce_mean(tf.cast(corr_pre, tf.float32))
        tf.summary.scalar(name="acc", tensor=acc)
        merged = tf.summary.merge_all()

        # 所有变量初始化
        init_var = tf.global_variables_initializer()
        # 模型保存准备
        saver = tf.train.Saver(max_to_keep=1)
        """
             模型开始训练
        """
        # tensorflow1 模型训练要先加载会话
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(logdir="./summary/", graph=sess.graph)
            sess.run(init_var)
            # 暂时不做减枝处理
            sess.run(opt, feed_dict={x: input_x, y: input_y})
            sess.run(acc, feed_dict={x: input_x, y: input_y})
            summary_result = sess.run(fetches=merged, feed_dict={x: input_x, y: input_y})
            writer.add_summary(summary=summary_result, global_step=i)
            saver.save(sess=sess, save_path="./models/lenet-5.ckpt", global_step=i + 1)
            print(acc)


if __name__ == '__main__':
    # 先限制tensorflow输出日志
    # TFTool(0)
    # 开始加载数据
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for i in range(len(gpus)):
                tf.config.experimental.set_memory_growth(gpus[i], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    mnist = input_data.read_data_sets("/home/fc/dataset/mnist/mnistGZ/", one_hot=True)
    # 加载模型
    cnn_recog_mnist = CNN(28, 28, 1, 784, 10)  # 图片长、宽、通道数
    # 模型训练
    batch_size = 50
    epochs = 10000
    for i in range(10000):
        batch = mnist.train.next_batch(batch_size)
        cnn_recog_mnist.train(batch[0], batch[1])
