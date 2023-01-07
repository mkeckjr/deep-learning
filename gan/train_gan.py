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
An implementation of training a vanilla GAN on MNIST in TensorFlow
"""

from __future__ import print_function

import argparse
import os

import numpy
import torch
from torch.nn import (
    BatchNorm1d,
    Linear,
    Flatten,
    LeakyReLU,
    Module,
    Sequential
)
import torchvision
from tqdm import tqdm

from . import basegan
from . import dump_gan_images


class MLPGenerator(Module):
    def __init__(
        self,
        latent_dim,
        output_shape,
        dense_shape=512,
        dtype=torch.float32,
        device=None
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        kwargs = {'dtype': dtype}
        if device is not None:
            kwargs['device'] = device

        flattened_shape = numpy.prod(output_shape)
        self.base_model = Sequential(
            Linear(latent_dim, dense_shape, **kwargs),
            BatchNorm1d(dense_shape, **kwargs),
            LeakyReLU(),
            Linear(dense_shape, dense_shape, **kwargs),
            BatchNorm1d(dense_shape, **kwargs),
            LeakyReLU(),
            Linear(dense_shape, dense_shape, **kwargs),
            BatchNorm1d(dense_shape, **kwargs),
            LeakyReLU(),
            Linear(dense_shape, flattened_shape, **kwargs)
        )

    def forward(self, x):
        vectorized = self.base_model(x)
        batch_size = vectorized.shape[0]
        output = torch.reshape(vectorized, (batch_size,) + self.output_shape)
        return output


class MLPDiscriminator(Module):
    def __init__(
        self,
        input_shape,
        dense_shape=512,
        dtype=torch.float32,
        device=None
    ):
        super().__init__()
        self.input_shape = input_shape

        kwargs = dict()
        kwargs['dtype'] = dtype
        if device is not None:
            kwargs['device'] = device

        flatten_shape = numpy.prod(input_shape)
        self.model = Sequential(
            Linear(flatten_shape, dense_shape, **kwargs),
            LeakyReLU(),
            Linear(dense_shape, dense_shape, **kwargs),
            LeakyReLU(),
            Linear(dense_shape, dense_shape, **kwargs),
            LeakyReLU(),
            Linear(dense_shape, 1, **kwargs)
        )

    def forward(self, x):
        flattened = torch.flatten(x, start_dim=1)
        return torch.sigmoid(self.model(flattened))


def train(train_dataloader,
          gan_model,
          n_epochs,
          device,
          test_dataloader=None,
          n_critic_iters_per_gen_update=10,
          outdir=None,
          save_frequency=10):
    """Training function for the WGAN-GP model.
    """
    latent_dim = gan_model.latent_dim
    batch_size = gan_model.batch_size

    for epoch in range(n_epochs):
        # shuffle the real data
        print('Epoch {}'.format(epoch))

        burn_in_epochs = 1

        sum_disc_loss = 0
        for j, (data, _) in enumerate(tqdm(train_dataloader)):
            data = data.to(device)
            disc_loss = gan_model.train_discriminator(data)
            sum_disc_loss += disc_loss.cpu()

            if (epoch > burn_in_epochs) and (j % n_critic_iters_per_gen_update == 0):
                for k in range(2):
                    gan_model.train_adversarial()

        if j > 0:
            print(f'{epoch}: Avergage disc loss: {sum_disc_loss / j}')
        else:
            print(f'{epoch}: No discriminative loss')

        if test_dataloader is not None:
            for data, _ in test_dataloader:
                data = data.to(device)
                real_output = gan_model.discriminate(data).cpu()
                break

            fake_samples = gan_model.generate(
                gan_model.generate_noise()
            )
            fake_output = gan_model.discriminate(fake_samples).cpu()

            print('Real range: [{}, {}], Fake range: [{}, {}]'.format(
                real_output.min(),
                real_output.max(),
                fake_output.min(),
                fake_output.max()))

        if epoch % save_frequency == 0:
            outname = os.path.join(outdir, f'{epoch:09d}.png')
            print('Saving images to {}'.format(outname))

            out_ckpt = os.path.join(outdir, f'{epoch:09d}.pt')
            gan_model.save(out_ckpt, epoch=epoch)

            dump_gan_images.render(outname,
                                   gan_model, # .generator,
                                   10)

    return


def main(mnist_data_root,
         batch_size,
         latent_dim,
         epochs,
         image_output_dir,
         save_frequency,
         device):

    generator = MLPGenerator(latent_dim, (1,28,28), device=device)
    discriminator = MLPDiscriminator((1,28,28), device=device)

    # get the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        mnist_data_root, train=True, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4,
    )

    test_dataset = torchvision.datasets.MNIST(
        mnist_data_root, train=False, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4,
    )

    gan_model = basegan.GenerativeAdversarialNetwork(
        latent_dim,
        batch_size,
        generator,
        discriminator,
        dtype=torch.float32,
        device=device
    )

    train(train_dataloader, # data.astype(torch.float32),
          gan_model,
          epochs,
          device,
          test_dataloader=test_dataloader,
          outdir=image_output_dir,
          save_frequency=save_frequency)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train a Wasserstein GAN with gradient penalty on MNIST',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_root',
                        help='MNIST data root.',
                        type=str)

    parser.add_argument('--batch-size',
                        help='Size of batches.',
                        type=int,
                        default=50)

    parser.add_argument('--latent-dim',
                        help='Size of latent dimension of generator.',
                        type=int,
                        default=100)

    parser.add_argument('--epochs',
                        help='Size of batches.',
                        type=int,
                        default=1000)

    parser.add_argument('--image-output-dir',
                        help='Output directory in which to store generated images.',
                        type=str,
                        default=None)

    parser.add_argument('--gpu-id',
                        help='GPU ID on which to run (None indicates a CPU device).',
                        type=int,
                        default=None)

    parser.add_argument('--save-freq',
                        help='Frequency with which to save output images/checkpoints.',
                        type=int,
                        default=None)

    args = parser.parse_args()

    if args.gpu_id is None:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu_id}')

    main(args.data_root,
         args.batch_size,
         args.latent_dim,
         args.epochs,
         args.image_output_dir,
         args.save_freq,
         device)
