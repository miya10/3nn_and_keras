import numpy as np
import pickle
from mnist import MNIST
from train import *
from function import *

"""
--主要変数の説明--
X:入力画像データのnumpy配列
Y:画像の正解ラベルのnumpy配列
image_size:入力画像の1辺のサイズ
batch_size:バッチサイズ
input_size:入力サイズ
hidden_size:隠れ数のノード数
output_size:分類ラベルの個数
epoch:エポック数
mini_batch:バッチサイズで指定された枚数の入力画像
t:正解ラベルをone hot vector表記した配列
y:ニューラルネットワークの出力
cel:クロスエントロピー誤差
correct_num:正解数
accuracy:予測精度
gamma, beta:batch normalizationのパラメータ
"""

image_size = 28
batch_size = 100
#input_size = 784
input_size = 3072
hidden_size = 200
output_size = 10
epoch = 20

# mnistデータの読み込み
def load_train_data():
    mndata = MNIST("../le4nn/")
    X, Y = mndata.load_training()
    X = np.array(X)
    X = X.reshape((X.shape[0], image_size*image_size))
    X = X / 255.0
    Y = np.array(Y)
    return X, Y

# mnistデータの読み込み
def load_test_data():
    mndata = MNIST("../le4nn/")
    X, Y = mndata.load_testing()
    X = np.array(X)
    X = X.reshape((X.shape[0], image_size*image_size))
    X = X / 255.0
    Y = np.array(Y)
    return X, Y

# CIFAR読み込み
def load_cifar_data():
    for i in range(5):
        with open("../cifar-10-batches-py/data_batch_"+str(i+1), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        tmp_X = np.array(dict[b'data'])
        tmp_X = tmp_X.reshape((tmp_X.shape[0],3*32*32))
        tmp_X = tmp_X / 255.0
        tmp_Y = np.array(dict[b'labels'])
        if i == 0:
            X = tmp_X
            Y = tmp_Y
        else:
            X = np.append(X, tmp_X, axis=0)
            Y = np.append(Y, tmp_Y, axis=0)
    return X, Y

def load_cifar_test_data():
    with open("../cifar-10-batches-py/test_batch", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = np.array(dict[b'data'])
    X = X.reshape((X.shape[0],3*32*32))
    X = X / 255.0
    Y = np.array(dict[b'labels'])
    return X, Y

# 正解率の計算
def calculate_accuracy(W1, b1, W2, b2, gamma, beta, mean_list, var_list):
    """saved_weight = np.load('dev_a4.npz')
    W1 = saved_weight['W1']
    b1 = saved_weight['b1']
    W2 = saved_weight['W2']
    b2 = saved_weight['b2']"""
    #X, Y = load_test_data()
    X, Y = load_cifar_test_data()
    #predict_y = predict(X, W1, b1, W2, b2)
    predict_y = predict_test(X, W1, b1, W2, b2, gamma, beta, mean_list, var_list)
    result = np.argmax(predict_y, axis=1)
    correct_num = np.sum(result == Y)
    accuracy = correct_num / len(Y)
    print('accuracy = %f' % (accuracy))

# メイン関数
def main():
    #X, Y = load_train_data()
    X, Y = load_cifar_data()
    sample_num = len(Y)
    std = np.sqrt(1 / input_size)
    W1 = np.random.normal(0, std, (input_size, hidden_size))
    W2 = np.random.normal(0, std, (hidden_size, output_size))
    b1 = np.random.normal(0, std, (1, hidden_size))
    b2 = np.random.normal(0, std, (1, output_size))
    adam_cache = (0, 0, 0, 0, 0, 0)
    gamma = 1.0
    beta = 0.0
    mean_list = []
    var_list = []
    repeat_num = len(Y) // batch_size
    for i in range(epoch):
        for j in range(repeat_num):
            x = X[j * batch_size : (j + 1) * batch_size]
            t = Y[j * batch_size : (j + 1) * batch_size]
            W1, b1, W2, b2, gamma, beta, adam_cache, mean_list, var_list = update_weight(x, t, W1, b1, W2, b2, gamma, beta, adam_cache, mean_list, var_list)
        y = predict(X, W1, b1, W2, b2)
        t = one_hot_vector(Y)
        cel = cross_entropy_loss(y, t)
        print('epoch %d/%d : CEL = %f' % (i+1, epoch, cel))
    calculate_accuracy(W1, b1, W2, b2, gamma, beta, mean_list, var_list)
    np.savez('np_saved', W1=W1, b1=b1, W2=W2, b2=b2)

if __name__ == '__main__':
    main()
    #calculate_accuracy()