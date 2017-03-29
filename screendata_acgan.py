#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
MNIST dataset. See https://arxiv.org/abs/1610.09585 for more details.

You should start to see reasonable images after ~5 epochs, and good images
by ~15 epochs. You should use a GPU, as the convolution-heavy operations are
very slow on the CPU. Prefer the TensorFlow backend if you plan on iterating, as
the compilation time can be a blocker using Theano.

Timings:

Hardware           | Backend | Time / Epoch
-------------------------------------------
 CPU               | TF      | 3 hrs
 Titan X (maxwell) | TF      | 4 min
 Titan X (maxwell) | TH      | 7 min

Consult https://github.com/lukedeo/keras-acgan for more information and
example output
"""
from __future__ import print_function

from collections import defaultdict

from six.moves import cPickle 

from PIL import Image

from six.moves import range

import keras.backend as K
from keras.datasets import mnist
from keras.layers.core import Activation
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np
import gzip
import math

np.random.seed(1337)

K.set_image_dim_ordering('tf')

#生成器
def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28) theano
	# label drawn from P_c, to image space (..., 28, 28, 1) tensorflow 
    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size, activation='tanh'))
    cnn.add(Dense(512 * 4 * 4, activation='tanh'))
    cnn.add(Reshape((4, 4, 512)))# tensorflow

    # upsample to (..., 8, 8)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(256, 5, 5, border_mode='same',
                          activation='tanh', init='glorot_normal'))

    # upsample to (..., 16, 16)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(128, 5, 5, border_mode='same',
                          activation='tanh', init='glorot_normal'))

    # take a channel axis reduction
    # upsample to (..., 32, 32)
    cnn.add(UpSampling2D(size = (2,2)))
    cnn.add(Convolution2D(64, 5, 5, border_mode='same',
                          activation='tanh', init='glorot_normal'))

    #upsample to (..., 64, 64)
    cnn.add(UpSampling2D(size = (2,2)))
    cnn.add(Convolution2D(32,5,5,border_mode = 'same',activation='tanh',init='glorot_normal'))
    
    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(2, latent_size,
                              init='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = merge([latent, cls], mode='mul')

    fake_image = cnn(h)

    return Model(input=[latent, image_class], output=fake_image)


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()
	
	#in theano, input_shape=(1,28,28)
	#in tensorflow, input_shape = (28,28,1)
    cnn.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2),
                          input_shape=(64, 64, 1)))
    cnn.add(Activation('tanh'))
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(Activation('tanh'))
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(256, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(Activation('tanh'))
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(512,3,3, border_mode = 'same', subsample=(2,2)))
    cnn.add(Activation('tanh'))
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(64, 64, 1))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(2, activation='softmax', name='auxiliary')(features)

    return Model(input=image, output=[fake, aux])
	
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
            img[ :, :, 0]
    return image

	
def load_data():
    f = open("screendata_ng.pkl",'rb')
    s_data = cPickle.load(f)
    s_label = cPickle.load(f)
    f = open("screendata_ok.pkl",'rb')
    s_data1 = cPickle.load(f)
    s_label1 = cPickle.load(f)
    return s_data,s_label,s_data1,s_label1


if __name__ == '__main__':

    # batch and latent size taken from the paper
    nb_epochs = 20
    batch_size = 100
    latent_size = 100
    classnum = 2

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # build the generator
    generator = build_generator(latent_size)
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                      loss='binary_crossentropy')

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model(input=[latent, image_class], output=[fake, aux])

    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # get our mnist data, and force it to be of shape theano(..., 1, 28, 28) or tensorflow(..., 28, 28, 1) with
    # range [-1, 1]
    #get data
    X_train,y_train,X_train_ok,y_train_ok= load_data()
    print(X_train.shape)
    print(y_train.shape)
    print(X_train_ok.shape)
    X_train = np.concatenate((X_train,X_train_ok))
    y_train = np.concatenate((y_train,y_train_ok))
    print(X_train.shape)
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    #X_train = np.expand_dims(X_train, axis=1)

    #X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    #X_test = np.expand_dims(X_test, axis=1)

    nb_train= X_train.shape[0]
	
    train_history = defaultdict(list)
    test_history = defaultdict(list)
	
    #reshape
    Xrow = 64
    X_train = np.reshape(X_train, (X_train.shape[0], Xrow, Xrow, 1))
    #X_test = np.reshape(X_test, (X_test.shape[0], Xrow, Xrow, X_test.shape[1]))
	
    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            progress_bar.update(index)
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))

            # get a batch of real images
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # sample some labels from p_c
            sampled_labels = np.random.randint(0, classnum, batch_size)
	    

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
	    
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((sampled_labels.shape[0], 1))], verbose=0)
				
            #print generated_images.shape
            if index % 5== 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save("./gene/"+str(epoch)+"_"+str(index)+".png")
	    

            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * batch_size + [0] * batch_size)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

	
            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))

            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, classnum, 2 * batch_size)

            # we want to train the genrator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size)

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((sampled_labels.shape[0], 1))], [trick, sampled_labels]))

        print('\nTesting for epoch {}:'.format(epoch +1))
"""
        # evaluate the testing loss here

        # generate a new batch of noise
        noise = np.random.uniform(-1, 1, (nb_test, latent_size))

        # sample some labels from p_c and generate images from them
        sampled_labels = np.random.randint(0, classnum, nb_test)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)

        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # make new noise
        noise = np.random.uniform(-1, 1, (2 * nb_test, latent_size))
        sampled_labels = np.random.randint(0, 10, 2 * nb_test)

        trick = np.ones(2 * nb_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights(
            'params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights(
            'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

        # generate some digits to display
        noise = np.random.uniform(-1, 1, (100, latent_size))

        sampled_labels = np.array([
            [i] * 10 for i in range(10)
        ]).reshape(-1, 1)

        # get a batch to display
        generated_images = generator.predict(
            [noise, sampled_labels], verbose=0)

        # arrange them into a grid
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(generated_images, 10)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(img).save(
            'plot_epoch_{0:03d}_generated.png'.format(epoch))

    cPickle.dump({'train': train_history, 'test': test_history},
                open('acgan-history.pkl', 'wb'))
"""
