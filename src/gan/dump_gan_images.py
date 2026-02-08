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

from __future__ import print_function

import argparse

import cv2
import keras.models
import numpy

def render(filename, gan, grid_size):
    latent_dim = gan.latent_dim
    z = numpy.random.normal(0,1,size=(grid_size*grid_size, latent_dim)).astype(numpy.float32)
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
