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


    def _build_random_weighted_average(self):
        """Create a random weighted average of two tensorflow tensors
        """
        shapes = [shp.value for shp in self.real_input.shape]

        random_values = tensorflow.random_uniform(
            shape=[self.batch_size,] + shapes[1:],
            minval=0., maxval=1.
        )

        return (random_values * self.real_input +
                (1-random_values) * self.generated_examples)


    def _build_gradient_penalty(self):
        """Build the gradient penalty tensor
        """

        weighted_samples = self._build_random_weighted_average()
        discriminator_sample_output = self.discriminator(weighted_samples)

        gradients = tensorflow.gradients(discriminator_sample_output,
                                         [weighted_samples])[0]
        
        normed_gradients = tensorflow.sqrt(
            tensorflow.reduce_sum(
                tensorflow.square(gradients),
                reduction_indices=[1])
        )
        
        gradient_penalty = tensorflow.reduce_mean((1.-normed_gradients)**2)
        self.gradient_penalty = self.penalty_weight*gradient_penalty


    def _build_discriminator_loss(self):
        """Build the discriminator training loss function from the outputs
        """
        self._build_gradient_penalty()

        self.discriminator_loss = (
            # loss for real examples
            -tensorflow.reduce_mean(
                self.discriminator_output[0]) +

            # loss for fake examples
            tensorflow.reduce_mean(
                self.discriminator_output[1]) +

            # gradient penalty
            self.gradient_penalty
        )
