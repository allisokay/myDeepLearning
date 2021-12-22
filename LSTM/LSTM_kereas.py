'''
@author：fc
@date：  2021/12/19
@contact：675435108@qq.com
'''
"""
文件内容&功能简要：
# LSTM的实现:进行空气污染预测
使用技术：keras
"""
from util.TFUitl import *
import pandas as pd
import math
import numpy as np

pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
from datetime import datetime
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

def parse(x):
    """
    format date
    :param x:
    :return:
    """
    return datetime.strptime(x, '%Y %m %d %H')


def get_clean_data():
    # parse_dates参数:将选中的列组合为日期格式，合并的列为以下划线连接成为新的列,
    # index_col=0将日期列作为索引，date_parser:处理parse_dates中的值
    dataset = pd.read_csv("../assets/BeiJingWeather(2010-2014)/raw.csv", parse_dates=[['year', 'month', 'day', 'hour']],
                          index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)  # inplace:是否在原参数上进行修改
    # 手动修改列名
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'  # 以日期为索引名
    dataset['pollution'].fillna(0, inplace=True)  # 修改数据
    dataset = dataset[24:]  # 丢弃前24个小时
    print(dataset.head(5))  # 取出前5行
    dataset.to_csv('../assets/BeiJingWeather(2010-2014)/clean.csv')


def draw_data():
    dataset = pd.read_csv("../assets/BeiJingWeather(2010-2014)/clean.csv", index_col=0, header=0)
    values = dataset.values
    groups = [0, 1, 2, 3, 5, 6, 7]  # 画图列
    i = 1
    pyplot.figure(figsize=(10, 10))  # 这里的figsize和subplot中值无关系
    for group in groups:
        pyplot.subplot(len(groups), 1, i)  # 八行一列第一个图
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i = i + 1
    pyplot.show()


"""
  将序列问题转换为有监督问题，根据前面24个小时的天气情况和污染，预测下一个小时的污染
"""


def series_to_supervised():
    """
     数据准备:对风向特征列进行编码，对所有特征列进行归一化处理，然后将数据集转化为有监督学习问题
     同时将需要预测的当前时刻（t）的天气条件特征移除
    """
    values = pd.read_csv("../assets/BeiJingWeather(2010-2014)/clean.csv", index_col=0, header=0).values
    # 整合编码方向
    encoder = LabelEncoder()
    # LabelEncoder.fit_transform:将风速（wnd_spd）特征转换为整数，每个特征值都有对应的整数
    values[:, 4] = encoder.fit_transform(values[:, 4])
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))  # 特征标准化
    scaled = scaler.fit_transform(values)
    n_vars = scaled.shape[1]
    df = pd.DataFrame(scaled)
    cols, names = list(), list()
    for i in range(1, 0, -1):  # 倒叙，n_in到0不含0️⃣
        cols.append(df.shift(i))  # 数据行上平移i个单位，最后i行会消失，平移不见了，平移新行填充Na
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # 预测序列
    for i in range(0, 1):
        cols.append(df.shift(-i))
        names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
    reframed = pd.concat(cols, axis=1)  # 将特征和标签混合
    reframed.columns = names
    reframed.dropna(inplace=True)  # 丢弃带有NaN值的行
    reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    print(reframed.head())
    return reframed, scaler


def train_test(data):
    """
     第一次修改：训练集、验证集和测试机划分=4:1
    :param data:
    :return:
    """
    values = data.values
    n_train_hours = 4*365 * 24
    train = values[:n_train_hours, :]
    pd.DataFrame(train,columns=['污染情况', '露点温度', '气温', '气压', '风向', '风速', '雪量', '雨量',"下一个时刻的污染情况"]).to_csv("../assets/BeiJingWeather(2010-2014)/train.csv",index=False)
    test = values[n_train_hours:, :]
    pd.DataFrame(test,columns=['污染情况', '露点温度', '气温', '气压', '风向', '风速', '雪量', '雨量',"下一个时刻的污染情况"]).to_csv("../assets/BeiJingWeather(2010-2014)/test.csv",index=False)
    # 输入和输出划分
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # 将形状转为为3维【样本数，时间点，特征】
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_y, test_X, test_y


def fit_network(train_X, train_y, test_X, test_y, scaler):
    model = Sequential()
    # 隐藏层有50个神经元
    model.add(LSTM(units=150, input_shape=(train_X.shape[1], train_X.shape[2])))  # 输出神经元大小50
    # 输出层1个神经元
    # model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    #  verbose：日志显示（0：不输出日志信息 *1：输出进度条，2：每个epoch输出一行记录）
    #  从训练集中取1%为验证集不加以训练
    history = model.fit(train_X, train_y,validation_split=0.1, verbose=2, epochs=200, batch_size=72, shuffle=False)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.legend()
    pyplot.show()
    # 准确率预测
    yhat = model.predict(test_X, verbose=0)  # 输出结果预测
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # 转换为实际值
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # rmse = math.sqrt(mean_squared_error(inv_y[1:], inv_yhat[:-1]))
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)


if __name__ == '__main__':
    # draw_data() # 数据集展示
    reframed, scaler = series_to_supervised()  # 将序列数据变成有监督数据
    # 划分训练集和测试集,一年数据为训练集，四年数据为测试集
    train_X, train_y, test_X, test_Y = train_test(reframed)
    # 搭建LSTM模型
    fit_network(train_X, train_y, test_X, test_Y, scaler)
