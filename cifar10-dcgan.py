# -*- coding:utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Reshape
from keras.layers.core import Activation  
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD,Adam
from keras.datasets import mnist
from six.moves import cPickle
import numpy as np
from PIL import Image
import argparse
import math
import gzip
import tarfile


def generator_model():
    model = Sequential()#单支线性模型
	#常用的全连接层
    model.add(Dense(input_dim=100, output_dim=1024)) 
	#激活函数层
    model.add(Activation('tanh'))
	#全连接层，表示输出是128*8*8维度的，
    model.add(Dense(128*8*8))
	#BatchNormalization层，该层在每个batch上将前一层的激活值重新规范化，使得其输出数据的均值接近于0，其标准差接近于1
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
	#增加一个reshape层，reshape层用来将输入shape转换成特定的shape
    model.add(Reshape((8, 8, 128), input_shape=(128*8*8,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))#卷积核的大小是3*3，output_channel =64，所以输出是64张featuremap
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Convolution2D(3,3,3,border_mode = 'same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()#单支线性模型
	#网络输入每张图片大小为单通道，大小是32*32，输出特征图是64个，卷积核的大小是3*3，其实就是有64个卷积核
    model.add(Convolution2D(
                        64, 3, 3,
                        border_mode='same',
                        input_shape=(32, 32, 3))) 
	#激活函数层
    model.add(Activation('tanh'))
	#最大池化层
    model.add(MaxPooling2D(pool_size=(2, 2)))#maxpooling层
	#卷积层，输出特征图为128，卷积核的大小为3*3，其实就是有128个卷积核
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))#maxpooling层
	#全连接展平，把多维的输入一维化Flatten不影响batch的大小
    model.add(Flatten())
	#全连接层，神经元的个数是1024
    model.add(Dense(1024))#隐藏层
    model.add(Activation('tanh'))
    model.add(Dense(1))#隐藏层
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = [32,32]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[ :, :, 0]
    return image

def load_cifar10(filename):
    fo = open(filename,'rb')
    dict1 = cPickle.load(fo)
    fo.close()
    return dict1

def deal_with_data(data):
    newdata = []
    img = np.zeros((32,32,3),dtype = data.dtype)
    for index in range(len(data)):
        X1 = data[index]
        X11 = X1[0:1024]
        image = X11.reshape(32,32)
        img[:,:,0] = image
        X11 = X1[1024:2048]
        image = X11.reshape(32,32)
        img[:,:,1] = image
        X11 = X1[2048:]
        image = X11.reshape(32,32)
        img[:,:,2] = image
        newdata.append(img)
    newdata = np.array(newdata)
    return newdata
    

def train(BATCH_SIZE):
    file1 = './cifar-10-python/cifar-10-batches-py/data_batch_1'
    dict_train_batch1 = load_cifar10(file1) #将data_batch文件读入到数据结构（字典）中
    X_train = dict_train_batch1.get('data')
    X_train = deal_with_data(X_train)
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
 #   channel = 3
 #   dd = int(math.sqrt(X_train.shape[1] / channel ))
 #   X_train = X_train.reshape(X_train.shape[0], dd, dd, channel) # tensorflow reshape to (..., 32, 32, 3)
    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
	#随机梯度下降
    d_optim = Adam(lr=0.0002, beta_1 = 0.5)
    g_optim = Adam(lr=0.0002, beta_1 = 0.5)
    generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))
    for epoch in range(20):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(noise,verbose = 0)
            if index % 5== 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save("./gene/"+str(epoch)+"_"+str(index)+".png")
                image = combine_images(image_batch)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save("./image/"+str(epoch)+"_"+str(index)+".png")

            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator')
        noise = np.zeros((BATCH_SIZE*20, 100))
        for i in range(BATCH_SIZE*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
