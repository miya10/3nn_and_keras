import numpy as np

"""
--主要変数の説明--
x:1次元の入力
t:正解ラベルをone hot vector表記した配列
dx, dW, db:勾配
"""

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU
def relu(x):
    return np.maximum(0, x)

# softmax function
def softmax(x):
    a = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - a)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# one-hot vector
def one_hot_vector(t):
    t = t.astype(np.int)
    t = np.eye(10)[t]
    return t

# cross entropy loss
def cross_entropy_loss(y, t):
    return - np.sum(t * np.log(np.maximum(y, 1e-7))) / y.shape[0]

# 誤差逆伝搬法(softmax)
def softmax_back(y, t):
    return (y - t) / y.shape[0]

# sigmoidの逆伝搬
def sigmoid_back(dx2, z):
    return (1 - z) * z * dx2

# ReLUの逆伝搬
def relu_back(dx2, z):
    return dx2 * np.where(z > 0, 1, 0)

# 勾配の計算
def grad(da, x, W, b):
    dx = np.dot(da, W.T)
    dW = np.dot(x.T, da)
    db = np.sum(da, axis=0)
    return dx, dW, db