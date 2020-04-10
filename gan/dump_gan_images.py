#!/usr/bin/python
from __future__ import print_function

import argparse

import cv2
import keras.models
import numpy

def render(filename, gan, grid_size):
    latent_dim = gan.generator_model.input_shape[1]
    z = numpy.random.normal(0,1,size=(grid_size*grid_size, latent_dim)).astype(numpy.float32)
    # z = gan.generate_noise()
    fake_batch = gan.generator(z)

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


def main(filename,
         generator_file,
         grid_size):

    generator = keras.models.load_model(generator_file)
    render(filename, generator, grid_size)

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Dump GAN images to image files as collages',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('generator',
                        help='Input generator, a Keras model filename.',
                        action='store')

    parser.add_argument('output_filename',
                        help='Name of output image file (use extension to indicate format).',
                        action='store')

    parser.add_argument('--grid-size',
                        help='Size of collage grid.',
                        type=int,
                        default=50)

    args = parser.parse_args()

    main(args.output_filename,
         args.generator,
         args.grid_size)
