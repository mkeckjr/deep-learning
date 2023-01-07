import argparse

import cv2
import numpy
import torch
import torchvision

from architectures.resnets import ResNet18, ResNetInput


class MNISTClassifier(torch.nn.Module):
    def __init__(self, backbone, flatten_dim, output_dim=10):
        super().__init__()

        self.backbone = backbone
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(flatten_dim, output_dim)

    def forward(self, x):
        return self.linear(self.flatten(self.backbone(x)))


def main(mnist_data_root, batch_size, epochs=100):
    device=torch.device('cuda:0')
    input_layer = ResNetInput(
        in_channels=1,
        conv_stride=1,
        pool_stride=1
    )
    model_backbone = ResNet18(
        in_channels=1, input_layer=input_layer
    ).float()

    # get the MNIST dataset
    dataset = torchvision.datasets.MNIST(
        mnist_data_root, train=True, download=True,
        transform=torchvision.transforms.ToTensor()
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=4
    )

    # get the model size for the flatten area
    data_point, target = dataset[0]
    data_point = data_point.unsqueeze(0)

    output_test = model_backbone(data_point)
    flatten_dim = output_test.shape[1] * output_test.shape[2] * output_test.shape[3]

    model = MNISTClassifier(
        model_backbone, flatten_dim, output_dim=10
    )
    model.train()
    model.to(device)

    # create a loss function
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.00001)

    for i in range(epochs):
        for j, (data, labels) in enumerate(dataloader):
            opt.zero_grad()
            output = loss(model(data.to(device)), labels.to(device))
            output.backward()
            opt.step()
            if j % 100 == 0:
                print(f'{i},{j} - Loss value: {float(output.cpu())}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a ResNet model on MNIST.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        'data_root',
        help='MNIST data root.',
        type=str,
    )
    parser.add_argument(
        '--batch-size',
        help='Batch size of training.',
        type=int,
        default=8
    )

    args = parser.parse_args()

    main(args.data_root, args.batch_size)

