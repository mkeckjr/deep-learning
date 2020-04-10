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


    def _build_adversarial_loss(self):
        """Build the adversarial loss tensor

        This private function should only be accessed by the instance. It builds
        the adversarial loss tensor, and assumes the adversarial model has 
        already been built.

        At the function's end, one should expect the following members to be
        defined:
            adversarial_loss: scalar tensorflow Tensor that represents the loss
                of the adversarial model on a batch
        """
        self.adversarial_loss = -tensorflow.reduce_mean(
            self.adversarial_output
        )

    def _build_discriminator_loss(self):
        """Build the discriminator training loss function from the outputs
        """
        self.discriminator_loss = (
            # loss for real examples
            -tensorflow.reduce_mean(
                self.discriminator_output[0]) +

            # loss for fake examples
            tensorflow.reduce_mean(
                self.discriminator_output[1])
        )


    def train_discriminator(self, session, real_batch):
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
