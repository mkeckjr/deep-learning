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
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.models import Model, Sequential


def compute_transpose_conv2d_size(prev_size, new_size):
    # get the scale factor for each of the rows/cols
    row_stride = new_size[1] // prev_size[1]
    col_stride = new_size[2] // prev_size[2]

    if 0 == row_stride or 0 == col_stride:
        raise RuntimeError('Conv2DTranspose {} -> {} impossible due to '
                           'zero stride'.format(prev_size, filter_size))

    if (new_size[1] == row_stride * prev_size[1] and
        new_size[2] == col_stride * prev_size[2]):
        # here we can use same padding and just leave
        return "same", (row_stride, col_stride), (1,1)

    # in this case we have to use "valid" padding, so we need to compute the
    # kernel size for each of the filters
    row_kernel_size = (new_size[1] - row_stride * (prev_size[1]-1))
    col_kernel_size = (new_size[2] - row_stride * (prev_size[2]-1))

    return "valid", (row_stride, col_stride), (row_kernel_size, col_kernel_size)


def construct_generator_old(output_rows,
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


def construct_generator_upsample(output_rows,
                                 output_cols,
                                 output_channels,
                                 latent_dim,
                                 layer_filters,
                                 kernel_size):
    generator_model = Sequential()

    prev_size = layer_filters[0]
    input_dense_size = prev_size[0]*prev_size[1]*prev_size[2]
    generator_model.add(Dense(input_dense_size,
                              activation="relu",
                              input_dim=latent_dim))

    generator_model.add(Reshape(prev_size))

    for filter_size in layer_filters[1:]:
        # get the scale factor for each of the rows/cols
        row_scale = filter_size[1] / float(prev_size[1])
        col_scale = filter_size[2] / float(prev_size[2])

        # add an upsampler
        generator_model.add(UpSampling2D(size=(row_scale, col_scale),
                                         data_format='channels_first'))
        generator_model.add(Conv2D(filter_size[0],
                                   kernel_size,
                                   padding='same',
                                   data_format='channels_first'))
        prev_size = filter_size

    # one final upsample/convolutoin to get to the output space
    row_scale = float(output_rows) / prev_size[1]
    col_scale = float(output_cols) / prev_size[2]

    generator_model.add(UpSampling2D(size=(row_scale, col_scale),
                                     data_format='channels_first'))
    generator_model.add(Conv2D(output_channels, kernel_size,
                               padding='same',
                               data_format='channels_first'))

    print('GENERATOR\n')
    generator_model.summary()
    print()

    ginput = Input(shape=(latent_dim,))
    goutput = generator_model(ginput)
    gmodel = Model(ginput, goutput)

    return gmodel

def construct_generator(output_rows,
                        output_cols,
                        output_channels,
                        latent_dim,
                        layer_filters):
                        # kernel_size):
    generator_model = Sequential()

    prev_size = layer_filters[0]
    input_dense_size = prev_size[0]*prev_size[1]*prev_size[2]
    generator_model.add(Dense(input_dense_size,
                              activation="relu",
                              input_dim=latent_dim))

    generator_model.add(Reshape(prev_size))

    for filter_size in layer_filters[1:]:
        # make sure we get the correct stride setting
        padding, strides, kernel_sizes = compute_transpose_conv2d_size(
            prev_size,
            filter_size)

        # add an upsampler
        generator_model.add(Conv2DTranspose(filter_size[0],
                                            kernel_sizes,
                                            strides=strides,
                                            padding=padding,
                                            data_format='channels_first'))

        generator_model.add(LeakyReLU(alpha=0.2))
        prev_size = filter_size

    # one final upsample/convolutoin to get to the output space
    padding, strides, kernel_sizes = compute_transpose_conv2d_size(
            prev_size,
            [output_channels, output_rows, output_cols])

    # add an upsampler
    generator_model.add(Conv2DTranspose(output_channels,
                                        kernel_sizes,
                                        strides=strides,
                                        padding=padding,
                                        data_format='channels_first'))

    print('GENERATOR\n')
    generator_model.summary()
    print()

    ginput = Input(shape=(latent_dim,))
    goutput = generator_model(ginput)
    gmodel = Model(ginput, goutput)

    return gmodel


def construct_discriminator(rows,
                            cols,
                            channels,
                            layer_filters,
                            kernel_size,
                            strides=2):
    """Construct a discriminator/critic for Wasserstein-style GANs

    Args:
        rows: size of input image rows
        cols: size of input image cols
        channels: number of channels in the input
        layers_filters: list of integers where the length indicates the number
            of convolutional layers to use, and the value of each element
            is the number of filters to apply at each layer.
        kernel_size: integer indicating the width/height of each square
            convolutional kernel
        strides: integer indicating the stride to take at each layer

    Returns:
        A Keras convolutional model with a 4D tensor as input for imagery and
            a single real output.
    """

    discriminator_model = Sequential()

    first = True
    output_filters = []

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

        new_layer = discriminator_model.layers[-1]
        output_filters.append(new_layer.output_shape[1:])

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
    return Model(dinput, doutput), output_filters


def main(generator_output_file,
         discriminator_output_file,
         rows,
         cols,
         channels,
         latent_dim,
         filter_sizes,
         kernel_shapes):

    # create the disriminator network
    discriminator, output_filters = construct_discriminator(
        rows, cols, channels,
        filter_sizes,
        kernel_shapes)

    # create the generator network, using the shape of the discriminator
    # network to cue the sizes of each layer
    generator = construct_generator(rows,
                                    cols,
                                    channels,
                                    latent_dim,
                                    list(reversed(output_filters)))
                                    # kernel_shapes)

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

    parser.add_argument('--filter-sizes',
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
         args.filter_sizes,
         args.kernel_shapes)
