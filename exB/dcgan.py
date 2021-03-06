import numpy as np
import keras
import math
import os
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.layers import *
from keras.models import Sequential, Model
from keras.datasets import mnist
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.backend import set_session, tensorflow_backend
from sklearn.metrics import confusion_matrix, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto(gpu_options=tf.GPUOptions(
    allow_growth=True,
    visible_device_list="2"))
set_session(tf.Session(config=config))

class DCGAN():
    def __init__(self):
        self.epochs = 100
        self.latent_dim = 100
        self.batch_size = 128
        self.img_row = 28
        self.img_col = 28
        self.channel = 1
        self.img_shape = (self.img_row, self.img_col, self.channel)
        optimizer = optimizers.Adam(lr=0.0002, beta_1=0.5)
        # build and compile generator model
        self.generator = self.generator_model()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        # build and compile discriminator model
        self.discriminator = self.discriminator_model()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        # build and compile combined model
        self.discriminator.trainable = False
        self.combined = self.combined_model(self.generator, self.discriminator)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def generator_model(self):
        model = Sequential()
        model.add(Dense(input_dim=self.latent_dim, output_dim=1024))
        model.add(ReLU())
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Reshape((7, 7, 128)))
        model.add(Conv2DTranspose(64, kernel_size=(5,5), strides=(1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2DTranspose(32, kernel_size=(5,5), strides=(2,2), padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2DTranspose(self.channel, kernel_size=(5,5), strides=(2,2),
            padding='same', activation='tanh'))
        return model

    def discriminator_model(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(5,5),
            input_shape=self.img_shape, padding='same'))
        model.add(LeakyReLU())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (5, 5), padding='same'))
        model.add(LeakyReLU())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(LeakyReLU())
        model.add(Dense(1, activation='sigmoid'))
        return model

    def combined_model(self, generator, discriminator):
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        return model
    
    def save_image(self, generated_images, epoch):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated_images.shape[1:3]
        image = np.zeros((height*shape[0], width*shape[1]),
                        dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
                img[:, :, 0]
        image = image * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save(
            'mnist_output/' + str(epoch) + ".png")

    def train(self):
        (X_train, y_train), (_, _) = mnist.load_data()
        X_train = (X_train.astype('float32') - 127.5) / 127.5
        X_train = X_train[:, :, :, None]
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))
        for epoch in range(self.epochs):
            print('epoch:', epoch)
            iterations = int(X_train.shape[0] / self.batch_size)
            for ite in range(iterations):
                image_batch = X_train[ite * self.batch_size : (ite + 1) * self.batch_size]
                noise = np.random.uniform(-1, 1, size=(self.batch_size, self.latent_dim))
                generated_imgs = self.generator.predict(noise)
                X = np.concatenate((image_batch, generated_imgs))
                y = [1] * self.batch_size + [0] * self.batch_size
                d_loss = self.discriminator.train_on_batch(X, y)
                g_loss = self.combined.train_on_batch(noise, valid)
                self.discriminator.trainable = True
            self.generator.save_weights('generator', True)
            self.discriminator.save_weights('discriminator', True)
            DCGAN.save_image(generated_imgs, epoch)
            

if __name__ == '__main__':
    DCGAN = DCGAN()
    DCGAN.train()