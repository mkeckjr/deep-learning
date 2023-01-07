import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        self.model = torchvision.models.vgg16(pretrained=True)

        self.features = [self.model.features[0:4],
                         self.model.features[4:9],
                         self.model.features[9:16],
                         self.model.features[16:23]]

    def 


if __name__ == '__main__':
    loss_function = VGGPerceptualLoss()

