import functools
import math
from typing import Tuple

import torch
from torch.nn import BatchNorm2d, Conv2d, Identity, Module, ReLU, Sequential
import torch.nn.functional

def pad_and_downsample(
    x: torch.Tensor,
    padding: Tuple[int],
    pad_value: float,
    downsample: bool = False
):
    """Function for handling padding/downsampling across ResBlocks

    Args:
        x: torch.Tensor to pad/downsample
        padding: a padding tuple; please see documentation of 
            torch.nn.functional.pad and reference the 'pad' keyword argument
        pad_value: floating point value with which to pad
        downsample: downsample spatially by 2 if True

    Returns:
        Padded and downsampled x
    """
    padded = torch.nn.functional.pad(x, pad=padding, value=pad_value)
    if downsample:
        padded = padded[:, :, ::2, ::2]

    return padded

class ResidualBlock(Module):
    """The original form of the simple, 2-convolution residual layer

    The layer performs two convolutions along the main path, optionally 
    downsampling via stride at the input layer. In the case the input layer does
    downsample, the number of input features are doubled. The pad_mode allows
    the user to pad the identity part of the operation with zeros or to use a
    convolution to match dimensions before addition.
    """
    def __init__(
        self,
        in_channels,
        out_channels=None,
        internal_channels=None,
        kernel_size=3,
        stride=1,
        dtype=float,
        pad_mode='zeros'
    ):
        assert pad_mode in ['zeros', 'conv']

        if out_channels is None:
            out_channels = in_channels

        if internal_channels is None:
            internal_channels = in_channels

        if pad_mode == 'zeros' and out_channels < in_channels:
            raise ValueError("Cannot use 'zeros' pad mode when reducing the "
                             "number of channels.")

        super().__init__()
        self.bn1 = BatchNorm2d(in_channels, dtype=dtype)
        self.bn2 = BatchNorm2d(in_channels, dtype=dtype)
        self.relu = ReLU()
        self.conv1 = Conv2d(
            in_channels, internal_channels,
            kernel_size=kernel_size,
            padding='same',
            dtype=dtype
        )

        if 1 == stride:
            out_padding = 'same'
        else:
            out_padding = (kernel_size // 2, kernel_size // 2)
        self.conv2 = Conv2d(
            internal_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=out_padding,
            dtype=dtype
        )

        if out_channels == in_channels and 1 == stride:
            self.padder = Identity()
            return

        # in this case, we either have differing numbers of channels, or we
        # have changing resolution, or both
        if pad_mode == 'zeros':
            pad_dims = out_channels - in_channels
            self.padder = functools.partial(
                pad_and_downsample,
                padding=(0, 0,          # padding for final dimension
                         0, 0,          # padding for 2nd-last dimension
                         0, pad_dims),  # padding on channels dim
                pad_value=0,
                downsample=stride != 1
            )
        else:
            if 1 == stride:
                self.padder = Conv2d(in_channels, out_channels, kernel_size=1)
            else:
                self.padder = Conv2d(in_channels, out_channels,
                                     kernel_size=1, stride=2)

        return

    @property
    def out_channels(self):
        return self.conv2.out_channels

    def forward(self, x):
        x_through = self.relu(self.bn1(x))
        x_through = self.conv1(x_through)
        x_through = self.relu(self.bn2(x_through))
        x_through = self.conv2(x_through)

        z = self.padder(x)

        return x_through + z


class ResidualStage(Module):
    def __init__(
        self,
        in_channels,
        num_blocks,
        input_mode='half',
        kernel_size=3,
        stride=1,
        dtype=float
    ):
        super().__init__()

        assert input_mode in ['half', 'same']
        if input_mode == 'half':
            in_stride = 2
            out_channels = 2 * in_channels
        else:
            in_stride = 1
            out_channels = in_channels

        self.blocks = [
            ResidualBlock(
                in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=in_stride,
                dtype=dtype
            )
        ]

        for i in range(num_blocks-1):
            self.blocks.append(
                ResidualBlock(
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dtype=dtype
                )
            )

        self.model = Sequential(*self.blocks)

    @property
    def out_channels(self):
        return self.blocks[-1].out_channels

    def forward(self, x):
        return self.model(x)
        
