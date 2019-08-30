#!/usr/bin/python
"""
An implementation of the weight-clipping Wasserstein GAN in Keras
"""

from __future__ import print_function

import argparse

from keras.layers import (Activation,
                          BatchNormalization,
                          Dense,
                          Flatten,
                          Input,
                          LeakyReLU,
                          Reshape,
                          )
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential


def compute_projection_size(sequential_layer_kernels,
                            sequential_layer_padding,
                            sequential_layer_strides,
                            output_rc):
    cur_layer_rc = [output_rc[0], output_rc[1]]
    for kernel, padding, strides in zip(
            sequential_layer_kernels[::-1],
            sequential_layer_padding[::-1],
            sequential_layer_strides[::-1]):
        if padding == 'valid':
            remove = [2*(kernel[0]//2), 2*(kernel[1]//2)]
        elif padding == 'same':
            remove = [0,0]

        cur_layer_rc[0] = (cur_layer_rc[0] - remove[0]) // strides[0]
        cur_layer_rc[1] = (cur_layer_rc[1] - remove[1]) // strides[1]

        print('( {}, {} )'.format(cur_layer_rc[0], cur_layer_rc[1]))

    return cur_layer_rc


def construct_generator(output_rows,
                        output_cols,
                        output_channels,
                        latent_dim,
                        layer_filters,
                        kernel_size,
                        n_reshape_channels=1,
                        strides=2):
    n_layers = len(layer_filters)
    
    sequential_kernels = [[kernel_size, kernel_size]] * n_layers
    sequential_strides = [[strides, strides]] * n_layers
    sequential_padding = ['same'] * n_layers

    projection_rows_cols = compute_projection_size(
        sequential_kernels,
        sequential_padding,
        sequential_strides,
        [output_rows, output_cols])

    generator_model = Sequential()

    # first step in DCGAN - make a dense layer mapping the input
    # shape to the size we want
    generator_model.add(Dense(projection_rows_cols[0] *
                              projection_rows_cols[1] *
                              n_reshape_channels,
                              activation="relu",
                              input_dim=latent_dim))

    # now reshape the output of that to be an image
    generator_model.add(Reshape([n_reshape_channels,
                                 projection_rows_cols[0],
                                 projection_rows_cols[1]]))

    # now add the remaining layers, which are strided transposed
    # convolutions
    for filter_size in layer_filters:
        generator_model.add(Conv2DTranspose(filter_size,
                                            kernel_size,
                                            strides=strides,
                                            padding='same',
                                            data_format='channels_first'))
        generator_model.add(BatchNormalization(momentum=0.8))
        generator_model.add(Activation('relu'))

    # add one final 1x1 convolution to get to the number of output
    # channels
    generator_model.add(Conv2DTranspose(output_channels,
                                        1, # kernel_size,
                                        # strides=strides,
                                        padding='same',
                                        data_format='channels_first'))
    generator_model.add(BatchNormalization(momentum=0.8))
    generator_model.add(Activation('tanh'))

    print('GENERATOR\n')
    generator_model.summary()
    print()

    ginput = Input(shape=(latent_dim,))
    goutput = generator_model(ginput)

    # add the noise input
    return Model(ginput, goutput)


def construct_discriminator(rows,
                            cols,
                            channels,
                            layer_filters,
                            kernel_size,
                            strides=2):

    discriminator_model = Sequential()

    first = True
    for filter_size in layer_filters:
        if first:
            discriminator_model.add(Conv2D(filter_size,
                                           kernel_size,
                                           strides=strides,
                                           padding='same',
                                           input_shape=(channels, rows, cols),
                                           data_format='channels_first'))
            first = False
        else:
            discriminator_model.add(Conv2D(filter_size,
                                           kernel_size,
                                           strides=strides,
                                           padding='same',
                                           data_format='channels_first'))
        discriminator_model.add(BatchNormalization(momentum=0.8))
        discriminator_model.add(LeakyReLU())

    # add a flatten + dense layer
    discriminator_model.add(Flatten())
    discriminator_model.add(Dense(1))

    # done-zo
    print('DISCRIMINATOR\n')
    discriminator_model.summary()
    print()

    dinput = Input(shape=(channels, rows, cols))
    doutput = discriminator_model(dinput)
    return Model(dinput, doutput)


def main(generator_output_file,
         discriminator_output_file,
         rows,
         cols,
         channels,
         latent_dim,
         generator_filter_sizes,
         discriminator_filter_sizes,
         kernel_shapes):
    
    # create the generator network
    generator = construct_generator(rows,
                                    cols,
                                    channels,
                                    latent_dim,
                                    generator_filter_sizes,
                                    kernel_shapes)


    # create the disriminator network
    discriminator = construct_discriminator(rows, cols, channels,
                                            discriminator_filter_sizes,
                                            kernel_shapes)

    generator.save(generator_output_file)
    discriminator.save(discriminator_output_file)

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Create discriminator/generator model files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('generator_output_file',
                        help='Output filename for generator model.',
                        action='store')

    parser.add_argument('discriminator_output_file',
                        help='Output filename for discriminator model.',
                        action='store')

    parser.add_argument('--rows',
                        help='Size of generated image rows',
                        action='store',
                        required=True,
                        type=int)

    parser.add_argument('--cols',
                        help='Size of generated image rows',
                        action='store',
                        required=True,
                        type=int)

    parser.add_argument('--latent-dim',
                        help='Size of input latent dimension.',
                        action='store',
                        type=int,
                        required=True)

    parser.add_argument('--generator-filter-sizes',
                        help='List of integers, used as filter sizes for the generator.',
                        nargs='+',
                        action='store',
                        type=int,
                        required=True)

    parser.add_argument('--discriminator-filter-sizes',
                        help='List of integers, used as filter sizes for the generator.',
                        action='store',
                        nargs='+',
                        type=int,
                        required=True)

    parser.add_argument('--channels',
                        help='Number of channels in generated images',
                        action='store',
                        type=int,
                        default=1)

    parser.add_argument('--kernel-shapes',
                        help='Shape of all KxK kernels in conv and transposed conv layers.',
                        action='store',
                        type=int,
                        default=3)


    args = parser.parse_args()

    main(args.generator_output_file,
         args.discriminator_output_file,
         args.rows,
         args.cols,
         args.channels,
         args.latent_dim,
         args.generator_filter_sizes,
         args.discriminator_filter_sizes,
         args.kernel_shapes)
