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
        self.generator = generator_model
        self.discriminator = discriminator_model

        self._build_adversarial_model()
        self._build_adversarial_loss()
        self._build_discriminator_model()
        self._build_discriminator_loss()

        self.compile_optimizers()


    def _build_adversarial_model(self):
        """Build the adversarial training tensors

        This private function should only be accessed by the instance. It builds
        the adversarial model, which includes creating the adversarial input to
        the generator and feeding that through the discriminator model.

        At the function's end, one should expect the following members to be
        defined:
            adversarial_input: tensorflow.keras.layers.Input placeholder for 
                input to the adversarial model (shape=(None,latent_dim))
            adversarial_output: tensorflow Tensor that represents the output
                of the adversarial model
        """

        # build the adversarial model
        self.adversarial_input = Input(shape=(self.latent_dim,),
                                       name='adversarial_input')
        self.adversarial_output = self.discriminator(
            self.generator(self.adversarial_input)
        )


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
            tensorflow.math.log(self.adversarial_output)
        )


    def _build_discriminator_model(self):
        """Build the discriminator training tensors

        At the end of this function, one should expect the following members to
        be defined:
            discriminator_real_input: Input placeholder for the real inputs to 
                the discriminator loss function
            discriminator_fake_input: Input placeholder for the real inputs to 
                the discriminator loss function
            discriminator_output: Context-specific output or list of outputs of 
                the discriminator during training as a result of executing the
                discriminator on both the real and fake inputs
        """
        self.real_input = Input(shape=self.input_shape,
                                name='real_input')
        self.fake_input = Input(shape=(self.latent_dim,),
                                name='fake_input')
        self.generated_examples = self.generator(self.fake_input)

        self.discriminator_output = [
            self.discriminator(self.real_input),
            self.discriminator(self.generated_examples)
        ]


    def _build_discriminator_loss(self):
        """Build the discriminator training loss function from the outputs
        """

        self.discriminator_loss = (
            # loss for real examples
            tensorflow.reduce_mean(
                tensorflow.math.log(self.discriminator_output[0])) +

            # loss for fake examples
            tensorflow.reduce_mean(
                tensorflow.math.log(1 - self.discriminator_output[1]))
        )


    def compile_optimizers(self):
        """Creates the optimizer(s) used to train the network
        """

        # train the adversarial model with only the generator weights
        self.adversarial_opt = tensorflow.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9)
        self.adversarial_minimizer_op = self.adversarial_opt.minimize(
            self.adversarial_loss,
            var_list=self.generator.weights
        )
    
        # train the discriminator model with only the discriminator weights
        self.discriminator_opt = tensorflow.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9
        )
        self.discriminator_minimizer_op = self.discriminator_opt.minimize(
            self.discriminator_loss,
            var_list=self.discriminator.weights
        )


    def generate_noise(self):
        """Generate a noise batch
        """
        # return numpy.random.normal(size=(batch_size,latent_dim))
        return numpy.random.random(size=(self.batch_size,
                                         self.latent_dim))


    def train_adversarial(self, session):
        """Do a single batch update to the generator via the adversarial loss

        Args:
            session: a tensorflow Session instance
        """
        z = self.generate_noise()
        feed_dict = { 'adversarial_input:0' : z }
        session.run(self.adversarial_minimizer_op,
                    feed_dict=feed_dict)


    def train_discriminator(self, session, real_batch):
        """Do a single batch update to the discriminator

        Args:
            session: a tensorflow Session instance
            real_batch: numpy array with shape congruent with the input to the
                discriminator
        """
        z = self.generate_noise()
        feed_dict = {
            'real_input:0' : real_batch,
            'fake_input:0' : z
        }

        session.run(
            self.discriminator_minimizer_op,
            feed_dict=feed_dict
        )


    def draw_generator_samples(self):
        """Draw a single batch worth of samples
        """
        z = self.generate_noise()
        imgs = generator.predict(z)
        return imgs

