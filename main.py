import numpy as np
import struct
import gzip
import os, sys

from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse


def load_mnist(path, mode):
    # 读取文件
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % mode)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % mode)
    with gzip.open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
    label_matrix = np.zeros((images.shape[0], 10))
    length = len(labels)
    for i in range(length):
        label_matrix[i][labels[i]] = 1
    return images, label_matrix


# 计算交叉熵
def cross_entropy_loss(y_predict, y_true):
    y_shift = y_predict - np.max(y_predict, axis=-1, keepdims=True)
    y_exp = np.exp(y_shift)
    y_probability = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
    loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  # 损失函数
    gradient = y_probability - y_true
    return loss, gradient


# 引入自定义实现类
# 激活函数
class ReLU():
    def __init__(self):
        pass

    def forward(self, input):
        self.output = np.maximum(0, input)
        return self.output

    def backward(self, next_gd, regular):
        gradient = np.where(np.greater(self.output, 0), next_gd, 0)
        return gradient

    def step(self, lr):
        pass

    def zero_grad(self):
        pass


class fc_layer():
    def __init__(self, input_size, output_size, name):
        self.Weight = np.random.randn(input_size, output_size) / 1000
        self.Bias = np.random.randn(output_size)
        self.name = name

    def forward(self, last_input):
        self.last_input = last_input
        output = np.dot(self.last_input, self.Weight) + self.Bias
        return output

    def backward(self, next_gd, regular):
        num = self.last_input.shape[0]
        gradient = np.dot(next_gd, self.Weight.T)
        dw = np.dot(self.last_input.T, next_gd)
        db = np.sum(next_gd, axis=0)
        self.dw = dw / num + regular * self.Weight
        self.db = db / num + regular * self.Bias
        return gradient

    def save(self, path):
        np.save(path + str(self.name) + '_Weight.npy', self.Weight)
        np.save(path + str(self.name) + '_Bias.npy', self.Bias)

    def load(self, path):
        self.Weight = np.load(path + str(self.name) + '_Weight.npy')
        self.Bias = np.load(path + str(self.name) + '_Bias.npy')

    def step(self, lr):
        self.Weight -= lr * self.dw
        self.Bias -= lr * self.db

    def zero_grad(self):
        self.dw = None
        self.db = None


# 优化器类
class optimizer():
    def __init__(self, layers, lr, regular=0.01):
        self.layers = layers
        self.length = len(layers)
        self.lr = lr
        self.regular = regular

    def backward(self, loss):
        for i in range(self.length - 1, -1, -1):
            loss = self.layers[i].backward(loss, self.regular)

    def step(self):
        for i in range(self.length):
            x = self.layers[i].step(self.lr)
        return x

    def zero_grad(self):
        for i in range(self.length):
            self.layers[i].zero_grad()


# 网络层类
class Sequential():
    def __init__(self, layers):
        self.layers = layers
        self.length = len(layers)

    def forward(self, x):
        for i in range(self.length):
            x = self.layers[i].forward(x)
        return x

    def save(self, path):
        for i in range(self.length):
            try:
                self.layers[i].save(path)
            except:
                pass

    def load(self, path):
        for i in range(self.length):
            try:
                self.layers[i].load(path)
            except:
                pass


def data_split(dataset, label, ratio):
    data_num = dataset.shape[0]
    size = int(data_num * ratio)
    idx = np.random.choice(data_num, size, replace=False)  # 随机取索引构建训练、验证集
    train_idx = list(set(range(data_num)) - set(idx))  # 相减取交集
    valid_data, valid_label = dataset[idx], label[idx]
    train_data, train_label = dataset[train_idx], label[train_idx]
    return train_data, train_label, valid_data, valid_label


# 随机批次，为后续训练使用
def random_batch(data, label, batch_size):
    data_num = data.shape[0]
    idx = np.random.choice(data_num, batch_size)
    return data[idx], label[idx]


#    构建模型和优化器
def build_model(input_size, hidden_size, output_size, lr=0.01, regular=0.01):
    layers = [fc_layer(input_size, hidden_size, 'fc1'), ReLU(), fc_layer(hidden_size, output_size, 'fc2')]
    model = Sequential(layers)
    model_optimizer = optimizer(layers, lr, regular=regular)
    return model, model_optimizer


def Accuracy(data, label, model):
    label_predict = model.forward(data)
    # 计算损失
    loss, _ = cross_entropy_loss(label_predict, label)
    #  计算模型准确率
    Accuracy = np.mean(np.equal(np.argmax(label_predict, axis=-1),
                                np.argmax(label, axis=-1)))
    return Accuracy, loss


def train(lr, regular, hidden_size):
    path = '.\\mnist_dataset'
    train_set, train_label = load_mnist(path, mode='train')
    test_set, test_label = load_mnist(path, mode='test')
    total_num = train_set.shape[0]
    batch_size = 256
    train_set, train_label, valid_set, valid_label = data_split(train_set, train_label, 0.2)
    steps = total_num // batch_size
    epoch = 7

    model, model_optimizer = build_model(train_set[0].shape[0], hidden_size, 10, lr, regular=regular)
    best_result = 0
    for i in range(epoch):
        for j in tqdm(range(steps)):

            data, label = random_batch(train_set, train_label, batch_size)
            label_predict = model.forward(data)
            loss, gradient = cross_entropy_loss(label_predict, label)
            model_optimizer.zero_grad()
            model_optimizer.backward(gradient)
            model_optimizer.step()

            if j % 100 == 0:
                Accuracy_train, loss_train = Accuracy(train_set, train_label, model)
                Accuracy_valid, loss_valid = Accuracy(valid_set, valid_label, model)
                Accuracy_test, loss_test = Accuracy(test_set, test_label, model)
                line = str(Accuracy_train) + ' ' + str(loss_train) + ' ' + str(Accuracy_valid) + ' ' + str(
                    loss_valid) + ' ' + str(Accuracy_test) + ' ' + str(loss_test) + '\r'

                with open('Training_Curve.txt', 'a+') as f:
                    f.write(line)

                if Accuracy_valid > best_result:
                    print("Epoch: {}, step: {}, loss: {}".format(i, j, loss))
                    print("Train Acc: {}; Valid Acc: {}".format(Accuracy_train, Accuracy_valid))
                    best_result = Accuracy_valid
                    model.save('./best_model_save/')


def mnist_test():
    path = '.\\mnist_dataset'
    test_set, test_label = load_mnist(path, mode='test')
    hidden_size = 512
    model, model_optimizer = build_model(test_set[0].shape[0], hidden_size, 10)
    model.load('./best_model_save/')
    Accuracy_test, loss_test = Accuracy(test_set, test_label, model)
    print("Test Acc: {}".format(Accuracy_test))
    return Accuracy_test

# 参数选择：循环遍历三种超参数的组合，最终得到相对较优的参数，用于后续的预测。
# best_result = 0
# best_combination = ''
# for learning_rate in [0.001, 0.005, 0.01, 0.05, 0.075]:
#     for regular in [0.005, 0.01, 0.05, 0.1]:
#         for hidden_size in [128, 256, 512]:
#             print(f"当前学习率为：{learning_rate}，regular为：{regular}，hidden_size为：{hidden_size}")
#             train(learning_rate, regular, hidden_size)
#             result = mnist_test()
#             line = str(learning_rate) + ' ' + str(regular) + ' ' + str(hidden_size) + ' ' + str(result) + '\r'
#             with open('Factor_Search.txt', 'a+') as f:
#                 f.write(line)
#             if result > best_result:
#                 best_result = result
#                 best_combination = line
# print(best_combination)
