from function import *

"""
--主要変数の説明--
x:入力の配列
y:計算結果
z:sigmoid活性化関数後の出力の一時保存用
t:one hot vector形式の正解ラベル
W1:入力層から隠れ層の重み
b1:入力層から隠れ層のバイアス
W2:隠れ層から出力層の重み
b2:隠れ層から出力層のバイアス
dW1, db1, dW2, db2:勾配
lr:学習率
rho:dropoutのパラメータ
beta1, beta2, eps:Adamのパラメータ
"""

batch_size = 100
lr = 0.01
rho = 0.5
alpha = 0.001
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

# 3層ニューラルネットワークの構成
def predict(x, W1, b1, W2, b2):
    y = np.dot(x, W1) + b1
    y = relu(y)
    # y = y * (1 - rho)
    y = np.dot(y, W2) + b2
    y = softmax(y)
    return y

# 3層ニューラルネットワークの構成
def predict_test(x, W1, b1, W2, b2, gamma, beta, mean_list, var_list):
    y = np.dot(x, W1) + b1
    mean = np.average(mean_list)
    var = np.average(var_list)
    # y = gamma / np.sqrt(var + eps) * y + (beta - gamma * mean / np.sqrt(var + eps))
    y = relu(y)
    # y = y * (1 - rho)
    y = np.dot(y, W2) + b2
    y = softmax(y)
    return y

# Adam
def adam(t, m, v, dW, W):
    t = t + 1
    m = beta1 * m + (1 - beta1) * dW
    v = beta2 * v + (1 - beta2) * dW * dW
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    W = W - alpha * m_hat / (np.sqrt(v_hat) + eps)
    return W, t, m, v

# Batch normalization
def batchnorm_forward(x, gamma, beta, mean_list, var_list):
    mean = np.mean(x, axis=0)
    var = np.mean((x - mean) ** 2, axis=0)
    mean_list.append(mean)
    var_list.append(var)
    x_hat = (x - mean) * 1.0 / np.sqrt(var + eps)
    out = gamma * x_hat + beta
    bn_cache = (x, x_hat, mean, var, gamma, beta)
    return out, bn_cache, mean_list, var_list

# Batch normalizationの逆伝搬
def batchnorm_backward(dout, cache):
    X, X_norm, mu, var, gamma, beta = cache

    N, D = X.shape

    X_mu = X - mu
    std_inv = 1. / np.sqrt(var + 1e-8)

    dX_norm = dout * gamma
    dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
    dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

    dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
    dgamma = np.sum(dout * X_norm, axis=0)
    dgamma = np.average(dgamma)
    dbeta = np.sum(dout, axis=0)
    dbeta = np.average(dbeta)
    return dX, dgamma, dbeta

# 逆伝搬
def back_propagation(x, y, z, t, W1, b1, W2, b2, dropout, bn_cache):
    da = softmax_back(y, t)
    dx2, dW2, db2 = grad(da, z, W2, b2)
    da2 = dx2 * dropout
    da2 = relu_back(dx2, z)
    # da2, dgamma, dbeta = batchnorm_backward(da2, bn_cache)
    dx1, dW1, db1 = grad(da2, x, W1, b1)
    return dW1, db1, dW2, db2, dgamma, dbeta

# 重みの更新
def update_weight(x, t, W1, b1, W2, b2, gamma, beta, adam_cache, mean_list, var_list):
    y = np.dot(x, W1) + b1
    y, bn_cache, mean_list, var_list = batchnorm_forward(y, gamma, beta, mean_list, var_list)
    z = relu(y)
    dropout = np.random.binomial(1, rho, size=(batch_size, 1))
    #y = z * dropout
    y = np.dot(z, W2) + b2
    y = softmax(y)
    t = one_hot_vector(t)
    dW1, db1, dW2, db2, dgamma, dbeta = back_propagation(x, y, z, t, W1, b1, W2, b2, dropout, bn_cache)
    t1, m1, v1, t2, m2, v2 = adam_cache
    W1, t1, m1, v1 = adam(t1, m1, v1, dW1, W1)
    b1 = b1 - lr * db1
    W2, t2, m2, v2 = adam(t2, m2, v2, dW2, W2)
    b2 = b2 - lr * db2
    gamma = gamma - lr * dgamma
    beta = beta - lr * dbeta
    adam_cache = (t1, m1, v1, t2, m2, v2)
    return W1, b1, W2, b2, gamma, beta, adam_cache, mean_list, var_list

"""def update_weight(x, t, W1, b1, W2, b2, gamma, beta, adam_cache, mean_list, var_list):
    y = np.dot(x, W1) + b1
    # y, bn_cache, mean_list, var_list = batchnorm_forward(y, gamma, beta, mean_list, var_list)
    bn_cache = 1
    z = relu(y)
    dropout = np.random.binomial(1, rho, size=(batch_size, 1))
    # y = z * dropout
    y = np.dot(z, W2) + b2
    y = softmax(y)
    t = one_hot_vector(t)
    dW1, db1, dW2, db2, dgamma, dbeta = back_propagation(x, y, z, t, W1, b1, W2, b2, dropout, bn_cache)
    t1, m1, v1, t2, m2, v2 = adam_cache
    W1, t1, m1, v1 = adam(t1, m1, v1, dW1, W1)
    b1 = b1 - lr * db1
    W2, t2, m2, v2 = adam(t2, m2, v2, dW2, W2)
    b2 = b2 - lr * db2
    # gamma = gamma - lr * dgamma
    # beta = beta - lr * dbeta
    return W1, b1, W2, b2, gamma, beta, adam_cache, mean_list, var_list"""