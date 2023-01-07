from torch.nn import BatchNorm2d, Conv2d, MaxPool2d, Module, ReLU, Sequential

from layers.resblock import ResidualBlock, ResidualStage


class ResNetInput(Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=64,
                 conv_kernel_size=7,
                 conv_stride=2,
                 pool_kernel_size=3,
                 pool_stride=2,
                 dtype=float):
        super().__init__()

        self.out_channels = out_channels

        self.bn = BatchNorm2d(in_channels, dtype=dtype)
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            dtype=dtype,
            padding=conv_kernel_size // 2
        )
        self.nonlinearity = ReLU()

        self.pool = MaxPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            padding=pool_kernel_size // 2
        )

    def forward(self, x):
        return self.pool(self.nonlinearity(self.conv(self.bn(x))))


class ResNetBase(Module):
    def __init__(
        self,
        in_channels=3,
        input_layer=None
    ):
        super().__init__()

        # create the initial mapping
        if input_layer is None:
            self.input_layer = ResNetInput(in_channels=in_channels)
        else:
            self.input_layer = input_layer

    def forward(self, x):
        return self.input_layer(x)


class ResNet18(ResNetBase):
    def __init__(
        self,
        in_channels=3,
        input_layer=None
    ):
        super().__init__(
            in_channels=in_channels,
            input_layer=input_layer,
        )

        self.stages = [ResidualStage(self.input_layer.out_channels, 2)]
        self.stages.append(
            ResidualStage(self.stages[-1].out_channels, 2)
        )
        self.stages.append(
            ResidualStage(self.stages[-1].out_channels, 2)
        )
        self.stages.append(
            ResidualStage(self.stages[-1].out_channels, 2)
        )

        self.model = Sequential(*self.stages)

    def forward(self, x):
        if self.input_layer is not None:
            x = self.input_layer(x)
        return self.model(x)
        
