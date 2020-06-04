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

import tensorflow

from .basegan import GenerativeAdversarialNetwork

class WGANGP(GenerativeAdversarialNetwork):
    """Wasserstein GAN with gradient penalty

    Implements the improved Wasserstein GAN training using a gradient penalty
    term to enforce the K-Lipschitz constraint.

      https://arxiv.org/abs/1704.00028
    """

    def __init__(self,
                 generator_model,
                 discriminator_model,
                 batch_size,
                 penalty_weight=10):
        """Initialize a Wasserstein GAN w/ gradient penalty

        Args:
            generator_model: tensorflow.keras.models.Model instance that defines
                the generator architecture
            discriminator_model: tensorflow.keras.models.Model instance that
                defines the discriminator architecture, which should take as
                input a tensor that is the shape of the generator's output
            batch_size: Integer, the size of the batches that will be used for
                training
            penalty_weight: Float, positive, the weight of the gradient
                penalty term
        """

        self.penalty_weight = penalty_weight
        super(WGANGP, self).__init__(generator_model,
                                     discriminator_model,
                                     batch_size)


    def random_weighted_average(self, real_batch, fake_batch):

        random_values = tensorflow.random.uniform(
            shape=[real_batch.shape[0]] + [1] * (len(real_batch.shape)-1),
            minval=0., maxval=1.
        )

        return (random_values * real_batch +
                (1-random_values) * fake_batch)


    def gradient_penalty(self, real_batch, fake_batch):
        rwa = self.random_weighted_average(real_batch, fake_batch)

        with tensorflow.GradientTape() as tape:
            tape.watch(rwa)
            disc_output = self.discriminator(rwa)

        gradients = tape.gradient(disc_output, [rwa])[0]
        normed_gradients = tensorflow.sqrt(
            tensorflow.reduce_sum(
                tensorflow.square(gradients),
                axis=[1,2,3])
        )

        gp = tensorflow.reduce_mean((1.-normed_gradients)**2)
        return self.penalty_weight*gp


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
        fake_batch = self.generator(z)

        with tensorflow.GradientTape() as tape:
            real_prediction = self.discriminator(real_batch)
            fake_prediction = self.discriminator(fake_batch)

            real_loss = -tensorflow.reduce_mean(real_prediction)
            fake_loss = tensorflow.reduce_mean(fake_prediction)

            # add a gradient penalty as well
            gradient_penalty = self.gradient_penalty(real_batch, fake_batch)

            loss = real_loss + fake_loss + gradient_penalty

        gradients = tape.gradient(loss, self.discriminator_model.weights)
        self.discriminator_opt.apply_gradients(
            zip(gradients, self.discriminator_model.weights)
        )
