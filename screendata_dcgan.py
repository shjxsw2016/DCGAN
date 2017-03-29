from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
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
#import cv2

#generator
def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dense(512*4*4))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #reshape to (..., 4, 4, 512)
    model.add(Reshape((4, 4, 512), input_shape=(4*4*512,)))
    #upsampling to (..., 8, 8, 512)
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, 3, 256, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #upsampling to (..., 16, 16, 256)
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, 3, 128, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #upsampling to (..., 32, 32, 128)
    model.add(UpSampling2D(size =(2,2)))
    model.add(Convolution2D(3, 3, 64, border_mode ='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #upsampling to (..., 64, 64, 64)
    model.add(UpSampling2D(size = (2, 2)))
    model.add(Convolution2D(3, 3, 32, model_mode = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #last Convoluton layer, (64, 64, 1)
    model.add(Convolution2D(3, 3, 1, border_mode = 'same'))
    model.add(Activation('tanh'))
    return model

#discriminator
def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(
                        64, 3, 3,
                        border_mode='same',
                        input_shape=(64, 64, 1)))
    model.add(Activation('tanh'))
    #subsampling to (..., 32,32,64)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('tanh'))
    #subsampling to (..., 16,16,128)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('tanh'))
    #subsampling to (..., 8,8,256)
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(512, 3, 3)
    model.add(Activation('tanh'))
    #subsampling to (..., 4,4,512)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

#generator and discriminator
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
    shape = [64,64]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

	
def load_data():
    f = open("screendata_ng.pkl",'rb')
    s_data = cPickle.load(f)
    s_label = cPickle.load(f)
    return s_data
	

def train(BATCH_SIZE):
    X_train= load_data()
    print X_train.shape
    #X_train = X_train.astype(np.float32)*255
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    dd = int(math.sqrt(X_train.shape[1]))
    X_train = X_train.reshape(X_train.shape[0], dd, dd, 1) # reshape to (..., 64,64,1)
    #print X_train[0]
    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    d_optim = Adam(lr=0.0002, beta_1 = 0.5)
    g_optim = Adam(lr=0.0002, beta_1 = 0.5)
    generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))
    for epoch in range(10):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)
           # print generated_images.shape
            if index % 5== 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save("./gene/"+str(epoch)+"_"+str(index)+".png")
            
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
    parser.add_argument("--path", type=str, default="screendata_ng.pkl")
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
