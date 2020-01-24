import numpy as np
import keras
from keras import datasets, models, layers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.backend import set_session, tensorflow_backend
from sklearn.metrics import confusion_matrix, accuracy_score
config = tf.ConfigProto(gpu_options=tf.GPUOptions(
    allow_growth=True,
    visible_device_list="2"))
set_session(tf.Session(config=config))

train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)
input_shape = (218, 178, 3)
num_classes = 2
batch_size = 32
train_generator = train_datagen.flow_from_directory('/home/iiyama/face/train/',
    target_size=(218,178),
    batch_size=batch_size,
    class_mode='categorical')
test_generator = test_datagen.flow_from_directory('/home/iiyama/face/test/',
    target_size=(218,178),
    batch_size=batch_size,
    class_mode='categorical') 

def cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu',
        input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    print(model.summary())
    return model

def train(model):
    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(), metrics=['acc'])
    # 学習．
    epochs = 20
    spe = 100 # 1 エポックあたりのバッチ数
    result = model.fit_generator(train_generator, steps_per_epoch = spe,
        epochs=epochs, validation_data=test_generator, validation_steps=30)
    history = result.history 
    model.save('smile2.h5')
    return model, history

def calculate_accuracy(model):
    loss, acc = model.evaluate_generator(test_generator, steps = 30)
    print ("loss:", loss)
    print ("accuracy:", acc)

def plot(history):
    fig = plt.figure()
    plt.plot(history['loss'], label='loss') # 教師データの損失
    plt.plot(history['val_loss'], label='val_loss') # テストデータの損失
    plt.legend()
    plt.savefig('result/smile_loss_history2.png')
    fig = plt.figure()
    plt.plot(history['acc'], label='acc') # 教師データでの精度
    plt.plot(history['val_acc'], label='val_acc') # テストデータでの精度
    plt.legend()
    plt.savefig("result/smile_loss_acc2.png") 

def main():
    model = cnn_model()
    model, history = train(model)
    calculate_accuracy(model)
    plot(history)

if __name__ == '__main__':
    main()