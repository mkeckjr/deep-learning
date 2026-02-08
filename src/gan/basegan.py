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
Class for representing the classic GAN model
"""

import numpy
import tensorflow

from tensorflow import keras
from keras.layers import Input

class GenerativeAdversarialNetwork(object):
    """Generative adversarial network interface and basic implementation

    Implements the classic generative adversarial network (GAN) and defines
    an interface for GANs in my software tree.

      https://arxiv.org/abs/1704.00028
    """

    def __init__(self,
                 generator_model,
                 discriminator_model,
                 batch_size,
                 *args,
                 **kwargs):
        """Initialize a classic GAN

        The GAN assumes two models are provided as input: (1) a generator which
        takes a 2-mode tensor as input that has shape (batch_size, latent_dim),
        and (2) a discriminator that takes a tensor the shape of the generator
        output and has output that is context dependent. In the vanilla case,
        the discriminator outputs a (batch_size,1) sigmod that indicates whether
        or not each element in the batch is real or fake.

        Args:
            generator_model: tensorflow.keras.models.Model instance that defines
                the generator architecture
            discriminator_model: tensorflow.keras.models.Model instance that
                defines the discriminator architecture, which should take as
                input a tensor that is the shape of the generator's output
        """

        self.latent_dim = generator_model.input_shape[-1]
        self.batch_size = batch_size
        self.input_shape = [shp
                            for shp in discriminator_model.input_shape
                            if shp is not None]

        self.generator_model = generator_model
        self.discriminator_model = discriminator_model

        self.create_optimizers()


    @tensorflow.function
    def generator(self, gen_input):
        return self.generator_model(gen_input)


    @tensorflow.function
    def discriminator(self, disc_input):
        return self.discriminator_model(disc_input)


    def create_optimizers(self):
        """Creates the optimizer(s) used to train the network
        """

        # train the adversarial model with only the generator weights
        self.adversarial_opt = tensorflow.keras.optimizers.Adam(
            learning_rate=1e-4,
            beta_1=0.5,
            beta_2=0.9)

        # train the discriminator model with only the discriminator weights
        self.discriminator_opt = tensorflow.keras.optimizers.Adam(
            learning_rate=1e-4,
            beta_1=0.5,
            beta_2=0.9
        )


    def generate_noise(self):
        """Generate a noise batch
        """
        return numpy.random.normal(
            size=(self.batch_size,
                  self.latent_dim)).astype(numpy.float32)


    def train_adversarial(self):
        """Do a single batch update to the generator via the adversarial loss
        """
        z = self.generate_noise()

        with tensorflow.GradientTape() as tape:
            loss = -tensorflow.reduce_mean(
                tensorflow.math.log(
                    self.discriminator(self.generator(z))
                )
            )

        gradients = tape.gradient(loss, self.generator_model.weights)
        self.adversarial_opt.apply_gradients(
            zip(gradients, self.generator_model.weights)
        )


    def train_discriminator(self, real_batch):
        """Do a single batch update to the discriminator

        Args:
            real_batch: numpy array with shape congruent with the input to the
                discriminator
        """
        z = self.generate_noise()

        with tensorflow.GradientTape() as tape:
            real_prediction = self.discriminator(real_batch)
            fake_prediction = self.discriminator(
                self.generator(z)
            )

            real_loss = tensorflow.reduce_mean(
                tensorflow.math.log(real_prediction))
            fake_loss = tensorflow.reduce_mean(
                tensorflow.math.log(1 - fake_prediction)
            )

            loss = real_loss + fake_loss

        gradients = tape.gradient(loss, self.discriminator_model.weights)
        self.discriminator_opt.apply_gradients(
            zip(gradients, self.discriminator_model.weights)
        )


    def draw_generator_samples(self):
        """Draw a single batch worth of samples
        """
        z = self.generate_noise()
        imgs = generator.predict(z)
        return imgs
