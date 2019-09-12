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
from keras.optimizers import RMSprop
import numpy
from progressbar import ProgressBar


def duplicated_render(filename, generator, grid_size):
    latent_dim = generator.input_shape[-1]
    z = numpy.random.random(size=(grid_size*grid_size, latent_dim))
    fake_batch = generator.predict(z)

    channels, height, width = fake_batch.shape[-3:]

    collage = numpy.zeros((height*grid_size, width*grid_size, channels))

    batch = 0
    mn = numpy.min(fake_batch)
    mx = numpy.max(fake_batch)
    dif = mx - mn
    for i in range(grid_size):
        for j in range(grid_size):
            collage[i*height:(i+1)*height,
                    j*width:(j+1)*width,:] = numpy.transpose(
                        (fake_batch[i*grid_size+j,...] - mn) / dif,
                        (1,2,0)
            )

    collage = numpy.squeeze(collage*255)
    cv2.imwrite(filename, collage.astype(numpy.uint8))


def wasserstein_loss(y_true, y_pred):
    """
    Loss function for a Wasserstein GAN.

    Args:
        y_true: Tensor indicating the true value for each input
        y_pred: Tensor indicating the predicted value for each input

    Returns: Wasserstein loss given the truth/prediction
    """
    return -keras.backend.mean(y_true * y_pred)


def train(data,
          generator,
          discriminator,
          combined,
          batch_size,
          n_epochs,
          test_data=None,
          n_critic_iterations_per_epoch=5):

    latent_dim = generator.input_shape[-1]
    n_real_images = data.shape[0]
    n_real_batches = n_real_images // batch_size
    inds = numpy.array(range(n_real_images))

    real_output_values = numpy.ones((batch_size,1))
    fake_output_values = -numpy.ones((batch_size,1))

    generator_loss = 1.

    for epoch in range(n_epochs):
        # shuffle the real data
        numpy.random.shuffle(inds)

        bar = ProgressBar()

        print('Epoch {}'.format(epoch))

        # update the critic repeatedly
        d_loss_total = 0

        if epoch < 25 or epoch % 100 == 0:
            n_critic_iterations = 100
        else:
            n_critic_iterations = n_critic_iterations_per_epoch

        for batch_ind in bar(range(n_critic_iterations)):
            modded_batch_ind = batch_ind % n_real_batches
            bstart = modded_batch_ind * batch_size

            real_batch = data[inds[bstart:bstart+batch_size],
                              ...]

            cur_batch_size = real_batch.shape[0]

            # make the same number of fake inputs
            z = numpy.random.random(size=(cur_batch_size, latent_dim))
            fake_batch = generator.predict(z)

            d_loss = discriminator.train_on_batch(
                numpy.concatenate((real_batch, fake_batch), axis=0),
                numpy.vstack((real_output_values[:cur_batch_size],
                              fake_output_values[:cur_batch_size]))
            )

            d_loss_total += d_loss

            # do the clipping thing
            clipped_weights = [numpy.clip(w, -0.01, 0.01)
                               for w in discriminator.get_weights()]
            discriminator.set_weights(clipped_weights)

        # update the generator only once at the end
        if epoch > 25:
            z = numpy.random.random(size=(cur_batch_size, latent_dim))
            generator_loss = combined.train_on_batch(z,
                                                     real_output_values[:cur_batch_size])

        print('d_loss_total = {}, d_loss = {}, g_loss = {}'.format(
            d_loss_total, d_loss,
            generator_loss))

        if test_data is not None:
            num_elements = test_data.shape[0]
            real_output = discriminator.predict(test_data)
            fake_output = combined.predict(
                numpy.random.random(size=(num_elements, latent_dim))
            )

            print('Real range: [{}, {}], Fake range: [{}, {}]'.format(
                real_output.min(),
                real_output.max(),
                fake_output.min(),
                fake_output.max()))

        generator.save('mygen.mdl')
        discriminator.save('mydisc.mdl')

        if epoch % 50 == 0:
            duplicated_render('wgan_images/{:09d}.png'.format(epoch),
                              generator,
                              10)

    return


def normalize_data(input_data):
    # make data fit in [-1,1] range
    mn = input_data.min()
    mx = input_data.max()

    output_data = 2.*((input_data - mn) / (mx-mn)) - 1.
    return output_data


def main(generator_file,
         discriminator_file,
         dataset_name,
         batch_size,
         epochs):

    generator = keras.models.load_model(generator_file)
    if len(generator.input_shape) != 2:
        raise ValueError('Input shape of generator model is not 2-mode')

    latent_dim = generator.input_shape[-1]
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

    data = normalize_data(data)
    test_data = normalize_data(test_data)

    print('Data range: [{},{}]'.format(data.min(), data.max()))
    print('Test data range: [{},{}]'.format(test_data.min(), test_data.max()))

    # compile the models; first compile the base discriminator
    discriminator.compile(loss=wasserstein_loss,
                          RMSprop(lr=0.00005))

    # okay, now turn off training of the discriminator to make a
    # combined loss
    discriminator.trainable = False
    combined_input = Input(shape=(1,1,latent_dim,))
    combined = Model(combined_input, discriminator(generator(combined_input)))
    combined.compile(loss=wasserstein_loss,
                     optimizer=RMSprop(lr=0.00005))

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
