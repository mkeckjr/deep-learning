# Deep Learning Implementations Library
# Copyright (C) 2020, Mark Keck
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
An implementation of training a vanilla GAN on MNIST in TensorFlow
"""

from __future__ import print_function

import argparse
import os

import keras.datasets
import numpy
from progressbar import ProgressBar
import tensorflow
from tensorflow import keras
from keras.layers import (
    BatchNormalization,
    Dense,
    Flatten,
    Input,
    LeakyReLU,
    Reshape
)
from keras.models import Model

from . import basegan
from . import dump_gan_images


def make_mnist_generator(output_shape=(1,28,28),
                         dense_shape=512,
                         latent_dim=100):
    """Doing this the simple way right now, I'll make it more general when it works.
    """

    z_input = Input((latent_dim,))

    dense1 = Dense(dense_shape, input_shape=(latent_dim,))(z_input)
    bn1 = BatchNormalization()(dense1)
    act1 = LeakyReLU()(bn1)

    dense2 = Dense(dense_shape, input_shape=(dense_shape,))(act1)
    bn2 = BatchNormalization()(dense2)
    act2 = LeakyReLU()(bn2)

    dense3 = Dense(dense_shape, input_shape=(dense_shape,))(act2)
    bn3 = BatchNormalization()(dense3)
    act3 = LeakyReLU()(bn3)

    dense4 = Dense(numpy.prod(output_shape),
                   input_shape=(dense_shape,),
                   activation='tanh')(act3)
    reshaped = Reshape(output_shape)(dense4)

    gen_model = Model([z_input], [reshaped])
    return gen_model


def make_mnist_discriminator(input_shape=(1,28,28),
                             dense_shape=512):

    d_input = Input(shape=input_shape)

    if len(input_shape) > 1:
        layer_input = Flatten()(d_input)
    else:
        layer_input = d_input

    dense1 = Dense(dense_shape,
                   input_shape=(numpy.prod(input_shape),))(layer_input)
    act1 = LeakyReLU()(dense1)

    dense2 = Dense(dense_shape,
                   input_shape=(dense_shape,))(act1)
    act2 = LeakyReLU()(dense2)

    dense3 = Dense(dense_shape,
                   input_shape=(dense_shape,))(act2)
    act3 = LeakyReLU()(dense3)

    output = Dense(1, input_shape=(dense_shape,))(act3)

    disc_model = Model([d_input], [output])
    return disc_model


def make_mnist_conv_discriminator(input_shape=(1,28,28),
                                  kernels):
    """Make an MNIST discriminator that is convolutional
    """

    d_input = Input(shape=input_shape)

    # keep it simple, 3 convolutional/max pooling layers
    # and an MLP layer at the end
    return None


def train(data,
          gan_model,
          n_epochs,
          test_data=None,
          n_critic_iters=5):
    """Training function for the WGAN-GP model.
    """

    latent_dim = gan_model.latent_dim
    batch_size = gan_model.batch_size

    n_real_images = data.shape[0]
    n_real_batches = n_real_images // batch_size
    inds = numpy.array(range(n_real_images))

    for epoch in range(n_epochs):
        # shuffle the real data
        numpy.random.shuffle(inds)
        bar = ProgressBar()

        print('Epoch {}'.format(epoch))

        # burn in hard early
        if epoch <= 25:
            n_batches_per_epoch = min(100,n_real_batches)
        else:
            n_batches_per_epoch = min(n_critic_iters, n_real_batches)

        for batch_i in bar(range(n_batches_per_epoch)):
            b_start = batch_i * n_real_batches
            b_end = b_start + batch_size

            # real and fake batch
            real_batch = data[inds[b_start:b_end], ...]
            z = numpy.random.random(size=(batch_size, latent_dim))

            if real_batch.shape[0] != batch_size:
                continue

            gan_model.train_discriminator(real_batch)

        if epoch > 5:
            gan_model.train_adversarial()

        if test_data is not None:
            num_elements = test_data.shape[0]
            real_output = gan_model.discriminator(test_data).numpy()
            # z = gan_model.generate_noise()
            fake_samples = gan_model.generator(
                numpy.random.random(size=(num_elements, latent_dim)).astype(numpy.float32)
            )
            fake_output = gan_model.discriminator(fake_samples).numpy()

            print('Real range: [{}, {}], Fake range: [{}, {}]'.format(
                real_output.min(),
                real_output.max(),
                fake_output.min(),
                fake_output.max()))

        gan_model.generator_model.save('mygen.mdl')
        gan_model.discriminator_model.save('mydisc.mdl')

        if epoch % 50 == 0:
            outdir = os.path.expanduser(
                '/home/deep-learning/gan/gan_images/{:09d}.png'.format(epoch))
            print('Saving images to {}'.format(outdir))

            dump_gan_images.render(outdir,
                                   gan_model, # .generator,
                                   10)

    return


def main(batch_size,
         epochs):

    generator = make_mnist_generator()
    latent_dim = generator.input_shape[-1]
    discriminator = make_mnist_discriminator()

    (x_train, y_train), (x_test,y_test) = keras.datasets.mnist.load_data()

    if len(x_train.shape) == 3:
        data = x_train[:,numpy.newaxis,...]
    else:
        data = x_train

    if len(x_test.shape) == 3:
        test_data = x_test[:,numpy.newaxis,...]
    else:
        test_data = x_test

    data = (data - 127.5) / 127.5
    test_data = (test_data - 127.5) / 127.5

    print('Data range: [{},{}]'.format(data.min(), data.max()))
    print('Test data range: [{},{}]'.format(test_data.min(), test_data.max()))


    gan_model = basegan.GenerativeAdversarialNetwork(
        generator,
        discriminator,
        batch_size)

    train(data.astype(numpy.float32),
          gan_model,
          epochs,
          test_data=test_data.astype(numpy.float32))

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train a Wasserstein GAN with gradient penalty on MNIST',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch-size',
                        help='Size of batches.',
                        type=int,
                        default=50)

    parser.add_argument('--epochs',
                        help='Size of batches.',
                        type=int,
                        default=1000)

    args = parser.parse_args()

    main(args.batch_size,
         args.epochs)
