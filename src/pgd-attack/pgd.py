import argparse
import glob
from typing import Callable

import cv2
import numpy
import PIL
import torch
import torchvision

def project_l2(noise, max_avg_pixel_dist):
    """Project a noise batch onto an L2 ball
    """
    max_dist = max_avg_pixel_dist * noise[0, ...].numel()
    with torch.no_grad():
        l2 = (noise ** 2).sum((1,2,3)).sqrt()
        scale = max_dist / l2
        scale[scale > 1] = 1.0
        noise *= scale

    return noise

def attack(img_batch: torch.Tensor,
           label_batch: torch.Tensor,
           model: torch.nn.Module,
           project_fn: Callable = project_l2,
           max_avg_pixel_dist: float = 0.01,
           max_steps: int = 100):
    """Perform a PGD attack given an image, label and model

    This function assumes that the image is normalized appropriately
    for the model.

    Args:
        img_batch: NCWH-shaped batch of images; N=1 is currently assumed
        label_batch: 1-mode tensor with N elements (again, N=1 assumed) 
            giving the ImageNet index label of the image batch
        model: an ImageNet classification model
        project_fn: the projection function used to project the noise back
            inside of a noise ball
        max_avg_pixel_dist: the maximum average pixel distance used to
            limit the size of the noise ball
        max_steps: the maximum number of PGD steps to takes

    Returns:
        PGD-perturbed version of the img_batch, NCWH format.
    """
    # ensure the model is in eval mode
    model.eval()
    ce = torch.nn.CrossEntropyLoss()

    noise = torch.zeros_like(img_batch, requires_grad=True).to('cuda:0')
    opt = torch.optim.SGD([noise], lr=0.01)

    x_last = img_batch

    for _ in range(max_steps):
        scores = model.forward(img_batch + noise)
        loss = -ce(scores, label_batch)
        loss.backward()
        opt.step()

        noise = project_fn(noise, max_avg_pixel_dist)
        opt.zero_grad()

        with torch.no_grad():
            x = img_batch + noise
            dif = ((x_last - x)**2).sum().sqrt().item()
            x_last = x
            noise_size = (noise**2).sum().sqrt().item()

        if dif < 1e-6:
            break

    return x_last

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Use a projected gradient descent (PGD) attack on an image.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'filename',
        type=str,
        help='Filename for image file to attack',
    )

    parser.add_argument(
        'true_class_index',
        type=int,
        help='The true class index for the ImageNet class of this example.'
    )

    parser.add_argument(
        '--pretrained-model',
        type=str,
        help='Named of pre-trained torchvision.models model to use for ImageNet classification.',
        default='resnet50'
    )

    parser.add_argument(
        '--output-filename',
        type=str,
        help='Output filename for the attacked video.',
        default=None
    )

    parser.add_argument(
        '--max-attack-steps',
        type=int,
        help='Maximum number of PGD steps to use in the attack.',
        default=100
    )

    parser.add_argument(
        '--max-avg-pixel-distance',
        type=float,
        help='Maximum average pixel distance allowable in the attack.',
        default=0.01
    )

    args = parser.parse_args()

    model_type = getattr(torchvision.models, args.pretrained_model)
    model = model_type(pretrained=True)
    model.eval()
    model.to('cuda:0')

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    img = cv2.imread(args.filename)
    img = (img / 255.)
    img = (img - numpy.array(imagenet_mean)) / numpy.array(imagenet_std)
    img_tensor = torch.tensor(img.transpose((2,0,1)).astype(numpy.float32))
    img_tensor = torch.unsqueeze(img_tensor, 0)
    img_tensor = img_tensor.to('cuda:0')

    label_tensor = torch.zeros(1, dtype=torch.long).to('cuda:0')
    label_tensor[0] = args.true_class_index

    new_img_tensor = attack(
        img_tensor, label_tensor, model,
        max_avg_pixel_dist=args.max_avg_pixel_distance,
        max_steps=args.max_attack_steps
    )

    new_img = numpy.squeeze(new_img_tensor.cpu().numpy()).transpose((1, 2, 0))
    new_img = new_img * numpy.array(imagenet_std) + numpy.array(imagenet_mean)
    new_img[new_img < 0] = 0
    new_img[new_img > 1] = 1
    new_img = (255 * new_img).astype(numpy.uint8)

    output = model.forward(img_tensor)
    orig_output = output.cpu().detach().numpy().flatten()

    output = model.forward(new_img_tensor)
    attack_output = output.cpu().detach().numpy().flatten()

    print(f'Original image class: {orig_output.argmax()}')
    print(f'Attacked image class: {attack_output.argmax()}')

    if args.output_filename:
        cv2.imwrite(args.output_filename, new_img)
