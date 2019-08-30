#!/usr/bin/python
"""
An implementation of the weight-clipping Wasserstein GAN in Keras
"""

from __future__ import print_function

import argparse
import os

import cv2
import keras.backend
import keras.datasets
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import numpy
from progressbar import ProgressBar

def wasserstein_loss(y_true, y_pred):
    """
    Loss function for a Wasserstein GAN.

    Args:
        y_true: Tensor indicating the true value for each input
        y_pred: Tensor indicating the predicted value for each input

    Returns: Wasserstein loss given the truth/prediction
    """
    return keras.backend.mean(y_true * y_pred)


def train(data,
          generator,
          discriminator,
          combined,
          batch_size,
          n_epochs,
          test_data=None,
          n_generator_sub_epochs=20):

    latent_dim = generator.input_shape[1]
    n_real_images = data.shape[0]
    inds = numpy.array(range(n_real_images))

    real_output_values = numpy.ones((batch_size,1))
    fake_output_values = -numpy.ones((batch_size,1))

    generator_loss = 1.

    for epoch in range(n_epochs):
        # shuffle the real data
        numpy.random.shuffle(inds)

        bar = ProgressBar()

        n_disc = 0

        print('Epoch {}'.format(epoch))

        # split up some batches
        for bstart in bar(range(0,n_real_images,batch_size)):
            real_batch = data[inds[bstart:bstart+batch_size],
                              ...]

            cur_batch_size = real_batch.shape[0]

            # make the same number of fake inputs
            z = numpy.random.normal(0,1,size=(cur_batch_size, latent_dim))
            fake_batch = generator.predict(z)

            d_loss = discriminator.train_on_batch(
                numpy.concatenate((real_batch, fake_batch), axis=0),
                numpy.vstack((real_output_values[:cur_batch_size],
                              fake_output_values[:cur_batch_size]))
            )

            n_disc += 0

            # do the dumb clipping thing
            clipped_weights = [numpy.clip(w, -0.01, 0.01)
                               for w in discriminator.get_weights()]
            discriminator.set_weights(clipped_weights)

            # and now do the generator
            # for k in range(n_generator_sub_epochs):
            if n_disc % n_generator_sub_epochs == 0:
                # double the batch size since the discriminator saw that many things
                z = numpy.random.normal(0,1,size=(cur_batch_size, latent_dim))
                generator_loss = combined.train_on_batch(z, real_output_values[:cur_batch_size])

        print('d_loss = {}, g_loss = {}'.format(
            d_loss,
            generator_loss))

        generator.save('mygen.mdl')
        discriminator.save('mydisc.mdl')

        if test_data is not None:
            num_elements = test_data.shape[0]

            real_output = discriminator.predict(test_data)
            fake_output = combined.predict(
                numpy.random.normal(0,1,size=(num_elements, latent_dim))
            )

            import ipdb
            ipdb.set_trace()

    return


def main(generator_file,
         discriminator_file,
         dataset_name,
         batch_size,
         epochs):

    generator = keras.models.load_model(generator_file)
    if len(generator.input_shape) != 2:
        raise ValueError('Input shape of generator model is not 2-mode')

    latent_dim = generator.input_shape[1]

    discriminator = keras.models.load_model(discriminator_file)

    dataset_module = getattr(keras.datasets, dataset_name)
    (x_train, y_train), (x_test,y_test) = dataset_module.load_data()

    if len(x_train.shape) == 3:
        data = x_train[:,numpy.newaxis,...]
    else:
        data = x_train

    if len(x_test.shape) == 3:
        test_data = x_test[:,numpy.newaxis,...]
    else:
        test_data = x_test


    # get the input sizes
    img_rows, img_cols = data.shape[-2:]

    # compile the models; first compile the base discriminator
    optimizer = Adam(lr=.001, decay=1e-6)
    discriminator.compile(loss=wasserstein_loss,
                          optimizer=optimizer)

    # okay, now turn off training of the discriminator to make a
    # combined loss
    discriminator.trainable = False

    combined_input = Input(shape=(latent_dim,))
    combined = Model(combined_input, discriminator(generator(combined_input)))
    combined.compile(loss=wasserstein_loss,
                     optimizer=optimizer)

    train(data, generator, discriminator, combined,
          batch_size, epochs, test_data=test_data)

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train a Wasserstein GAN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('generator',
                        help='Input generator, a Keras model filename.',
                        action='store')

    parser.add_argument('discriminator',
                        help='Input discriminator, a Keras model filename.',
                        action='store')

    parser.add_argument('data_input_file',
                        help='Input NPZ file.',
                        action='store')

    parser.add_argument('--batch-size',
                        help='Size of batches.',
                        type=int,
                        default=50)

    parser.add_argument('--epochs',
                        help='Size of batches.',
                        type=int,
                        default=1000)

    args = parser.parse_args()

    main(args.generator,
         args.discriminator,
         args.data_input_file,
         args.batch_size,
         args.epochs)
