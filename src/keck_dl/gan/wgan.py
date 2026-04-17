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
Class for representing the WGAN-GP model
"""

import numpy
import tensorflow

from .basegan import GenerativeAdversarialNetwork

class WGAN(GenerativeAdversarialNetwork):
    """Wasserstein GAN

    Implements the Wasserstein GAN training which uses weight clipping to
    enforce the Lipschitz constaint.

      https://arxiv.org/abs/1701.07875
    """

    def __init__(self,
                 generator_model,
                 discriminator_model,
                 batch_size,
                 clip=0.01):
        """Initialize a Wasserstein GAN w/ gradient penalty

        Args:
            generator_model: tensorflow.keras.models.Model instance that defines
                the generator architecture
            discriminator_model: tensorflow.keras.models.Model instance that
                defines the discriminator architecture, which should take as
                input a tensor that is the shape of the generator's output
            batch_size: Integer, the size of the batches that will be used for
                training
            clip: Positive float, clip weights in the discriminator outside
                this range
        """

        self.clip = clip
        super(WGAN, self).__init__(generator_model,
                                   discriminator_model,
                                   batch_size)


    def train_adversarial(self):
        """Do a single batch update to the generator via the adversarial loss
        """
        z = self.generate_noise()
        with tensorflow.GradientTape() as tape:
            loss = -tensorflow.reduce_mean(
                self.discriminator(self.generator(z))
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

            real_loss = -tensorflow.reduce_mean(real_prediction)
            fake_loss = tensorflow.reduce_mean(fake_prediction)

            loss = real_loss + fake_loss

        gradients = tape.gradient(loss, self.discriminator_model.weights)
        self.discriminator_opt.apply_gradients(
            zip(gradients, self.discriminator_model.weights)
        )

        clipped_weights = [numpy.clip(w, -self.clip, self.clip)
                           for w in self.discriminator_model.get_weights()]
        self.discriminator_model.set_weights(clipped_weights)
