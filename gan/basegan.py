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

from typing import Union

import numpy
import torch
import torch.nn


def test_gradients_for_nan(parameters):
    params = [param for param in parameters]
    for param in params:
        if torch.isnan(param.grad).any():
            import ipdb; ipdb.set_trace()


class GenerativeAdversarialNetwork:
    """Generative adversarial network interface and basic implementation

    Implements the classic generative adversarial network (GAN) and defines
    an interface for GANs in my software tree.

      https://arxiv.org/abs/1704.00028
    """

    MIN_CLASSIFICATION_SCORE = 1e-8

    def __init__(self,
                 latent_dim: int,
                 batch_size: int,
                 generator_model: torch.nn.Module,
                 discriminator_model: torch.nn.Module,
                 dtype: torch.dtype=None,
                 device: Union[torch.device,str]=None):
        """Initialize a classic GAN

        The GAN assumes two models are provided as input: (1) a generator which
        takes a 2-mode tensor as input that has shape (batch_size, latent_dim),
        and (2) a discriminator that takes a tensor the shape of the generator
        output and has output that is context dependent. In the vanilla case,
        the discriminator outputs a (batch_size,1) sigmod that indicates whether
        or not each element in the batch is real or fake.

        Args:
            latent_dim: size of the latent space
            generator_model: torch.nn.Module instance that defines the generator,
                which takes input Tensor of size (batch_size, latent_dim) and outputs
                (batch_size,) + output_dim of the data
            discriminator_model: torch.nn.Module instance that the discriminator
                which should take as input a tensor that is the shape of the 
                generator's output and provides a (batch_size,1) output where that
                output is between 0 and 1 (e.g. sigmoid)
        """
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        create_keys = dict()

        if dtype is not None:
            create_keys['dtype'] = dtype

        if device is not None:
            create_keys['device'] = device

        # get the input shape to the discriminator by pushing something through
        # the generator; using batch size of 2 to deal with batch norm
        x_in = torch.ones((2, latent_dim), **create_keys)
        y = generator_model(x_in)

        self.input_shape = y.shape[1:]
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.sampler = torch.distributions.normal.Normal(
            torch.tensor([0.0], **create_keys),
            torch.tensor([1.0], **create_keys)
        )
        self.MIN_CLASSIFICATION_TENSOR = torch.tensor(
            self.MIN_CLASSIFICATION_SCORE, **create_keys
        )
        self._create_optimizers()

    def generate(self, gen_input: torch.Tensor):
        """Run noise through the generator to create images.

        Args:
            gen_input: a torch.Tensor that is (batch_size, latent_dim) created
                by calling generate_noise()

        Returns:
            A torch.Tensor with shape (batch_size, C, W, H) containing generated 
                images
        """
        return self.generator_model(gen_input)

    def discriminate(self, disc_input):
        """Run a batch of images through discriminator

        Args:
            disc_input: torch.Tensor with shape (batch_size, C, H, W) of images

        Returns:
            A torch.Tensor with shape (batch_size,) classifying each instance in 
                the batch as real (value 1.0) or fake (value 0.0)
        """
        return self.discriminator_model(disc_input)

    def _create_optimizers(self):
        """Creates the optimizer(s) used to train the network
        """
        # train the adversarial model with only the generator weights
        self.adversarial_opt = torch.optim.Adam(
            self.generator_model.parameters(),
            lr=1e-4, betas=(0.5, 0.9)
        )

        # train the discriminator model with only the discriminator weights
        self.discriminator_opt = torch.optim.Adam(
            self.discriminator_model.parameters(),
            lr=1e-4, betas=(0.5, 0.9)
        )

    def generate_noise(self, batch_size: int=None):
        """Generate a noise batch

        Args:
            batch_size: int indicating the batch size of noise to create

        Returns: 
            A torch.Tensor with shape (batch_size, latent_dim) for passing 
                through the generator network.
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        return self.sampler.sample(
            sample_shape=(batch_size, self.latent_dim)
        ).squeeze(-1)

    def safe_log(self, scores):
        """Safely take the log of a tensor of real numbers.

        This function ensures that we do not take the log of zero.

        Args:
            scores: torch.Tensor of values of which we wish to take the log.

        Returns:
            torch.log(torch.maximum(scores, torch.tensor(epsilon)))
        """
        return torch.log(
            torch.maximum(scores, self.MIN_CLASSIFICATION_TENSOR)
        )

    def train_adversarial(self):
        """Do a single batch update to the generator via the adversarial loss

        Returns:
            torch.Tensor loss function 
        """
        z = self.generate_noise()

        classifications = self.discriminator_model(self.generator_model(z))
        loss = -torch.mean(self.safe_log(classifications))
        self.adversarial_opt.zero_grad()
        loss.backward()        
        self.adversarial_opt.step()

        return loss

    def train_discriminator(self, real_batch):
        """Do a single batch update to the discriminator

        Args:
            real_batch: numpy array with shape congruent with the input to the
                discriminator
        """
        z = self.generate_noise()
        real_prediction = self.discriminator_model(real_batch)
        fake_images = self.generator_model(z)
        fake_prediction = self.discriminator_model(fake_images)
 
        real_loss = -torch.mean(self.safe_log(real_prediction))
        fake_loss = -torch.mean(self.safe_log(1 - fake_prediction))

        loss = real_loss + fake_loss
        self.discriminator_opt.zero_grad()
        loss.backward()
        self.discriminator_opt.step()
        return loss

    def draw_generator_samples(self):
        """Draw a single batch worth of samples
        """
        return self.generator_model(self.generate_noise())

    def save(self, filename, **kwargs):
        """Save a checkpoint
        """

        outdict = {
            'generator_state_dict': self.generator_model.state_dict(),
            'adversarial_opt_state_dict': self.adversarial_opt.state_dict(),
            'discriminator_state_dict': self.discriminator_model.state_dict(),
            'discriminator_opt_state_dict': self.discriminator_opt.state_dict(),
        }

        outdict.update(kwargs)
        torch.save(outdict, filename)

    def load(self, ckpt_file):
        """Load a checkpoint
        """
        checkpoint = torch.load(ckpt_file)

        self.generator_model.load_state_dict(checkpoint['generator_state_dict'])
        self.adversarial_opt.load_state_dict(checkpoint['adversarial_opt_state_dict'])

        self.discriminator_model.load_state_dict(checkpoint['discriminator_state_dict'])
        self.discriminator_opt.load_state_dict(checkpoint['discriminator_opt_state_dict'])

        return checkpoint
