import numpy
import torch

from layers.resblock import ResidualBlock, ResidualStage


def test_resblock_interface():
    # push a single tensor through a pair of resblocks
    block1 = ResidualBlock(
        3,  # in_channels,
        out_channels=64
    )

    block2 = ResidualBlock(
        64,  # in_channels,
        stride=2
    )

    im = numpy.ones((2,3,224,224), dtype=float)
    tensor_im = torch.from_numpy(im)

    x = block1(tensor_im)
    y = block2(x)

    assert y.shape[2] == 112 and y.shape[3] == 112


def test_resstage_interface():
    im = numpy.ones((2,64,224,224))
    tensor_im = torch.from_numpy(im)

    stage = ResidualStage(64, 3)

    y = stage(tensor_im)

    assert y.shape[1:] == (128, 112, 112)
