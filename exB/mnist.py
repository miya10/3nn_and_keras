import numpy as np
import keras
from keras import datasets, models, layers
from keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.backend import set_session, tensorflow_backend
from sklearn.metrics import confusion_matrix, accuracy_score

config = tf.ConfigProto(gpu_options=tf.GPUOptions(
    allow_growth=True,
    visible_device_list="2"))
set_session(tf.Session(config=config))

img_rows, img_cols = 28, 28 # 画像サイズは 28x28
num_classes = 10 # クラス数
(X, Y), (Xtest, Ytest) = keras.datasets.mnist.load_data() # 訓練用とテスト（兼 validation）用のデータを取得
X = X.reshape(X.shape[0], img_rows, img_cols, 1) # X を (画 像 ID，28, 28, 1) の 4 次元配列に変換
Xtest = Xtest.reshape(Xtest.shape[0], img_rows, img_cols,1)
X = X.reshape(X.shape[0], 784)
Xtest = Xtest.reshape(Xtest.shape[0], 784)
X = X.astype('float32') / 255.0 # 各画素の値を 0～1 に正規化
Xtest = Xtest.astype('float32') / 255.0
input_shape = (img_rows, img_cols, 1)
Y = keras.utils.to_categorical(Y, num_classes) # one-hot-vector へ変換
Ytest1 = keras.utils.to_categorical(Ytest, num_classes)

def cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu',
        input_shape=input_shape, padding='same'))
    model.add(layers.Conv2D(64,(3,3),activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(num_classes, activation='softmax'))
    print(model.summary())
    return model

def my_model():
    model = models.Sequential()
    """model.add(layers.Dense(5, activation='sigmoid', input_shape=(784,)))
    model.add(layers.Dense(10, activation='softmax'))
    print(model.summary())"""
    model.add(layers.Dense(512, input_shape=(784,)))
    model.add(layers.Activation('relu'))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2024))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(2024))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(2024))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(2024))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(2024))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(2024))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(2024))
    model.add(layers.Activation('relu'))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10))
    model.add(layers.Activation('softmax'))
    return model

def train(model):
    """model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(), metrics=['acc'])"""
    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer='adam', metrics=['acc'])
    epochs = 20
    batch_size = 100
    result = model.fit(X, Y, batch_size=batch_size,
        epochs=epochs, validation_data=(Xtest, Ytest1))
    history = result.history
    return model, history

def predict(model):
    # Xtest に対してクラス識別．
    pred = model.predict_classes(Xtest)
    # 混同行列 各行が正解のクラス，各列が推定されたクラスに対応
    print (confusion_matrix(Ytest, pred, labels=np.arange(10)))
    # 正答率の算出
    print (accuracy_score(Ytest, pred)) 

def plot(history):
    fig = plt.figure()
    plt.plot(history['loss'], label='loss') # 教師データの損失
    plt.plot(history['val_loss'], label='val_loss') # テストデータの損失
    plt.legend()
    plt.savefig('result/過学習loss.png')
    fig = plt.figure()
    plt.plot(history['acc'], label='acc') # 教師データでの精度
    plt.plot(history['val_acc'], label='val_acc') # テストデータでの精度
    plt.legend()
    plt.savefig("result/過学習_acc.png") 

def main():
    #model = cnn_model()
    model = my_model()
    model, history = train(model)
    predict(model)
    plot(history)

if __name__ == '__main__':
    main()