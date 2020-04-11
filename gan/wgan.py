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


    def train_discriminator_old(self, session, real_batch):
        """Do a single batch update to the discriminator

        This implementation calls the base class implementation and then clips
        the gradients.

        Args:
            session: a tensorflow Session instance
            real_batch: numpy array with shape congruent with the input to the
                discriminator
        """

        super(WGAN, self).train_discriminator(
            session, real_batch)

        clipped_weights = [numpy.clip(w, -self.clip, self.clip)
                           for w in self.discriminator.get_weights()]
        self.discriminator.set_weights(clipped_weights)


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
