'''
Neural network architectures for dense prediction via CNN-based image and/or video models.
Created by Basile Van Hoorick for TCOW.
'''

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'model/'))

from __init__ import *

# Library imports.
import os
import sys
import timm


# NOTE: Not used in my augs, BUT used in most pretrained models.
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/constants.py
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DenseResNet(torch.nn.Module):

    def __init__(self, logger, timm_name, pretrained, frame_height, frame_width, patch_dim,
                 in_channels):
        super().__init__()
        self.logger = logger
        self.timm_name = timm_name
        self.pretrained = pretrained
        # Frame size.
        self.Hf = frame_height
        self.Wf = frame_width
        # Number of patches.
        self.Ho = frame_height // patch_dim
        self.Wo = frame_width // patch_dim
        # Patch size.
        self.ho = patch_dim
        self.wo = patch_dim
        # Number of channels.
        self.Ci = in_channels

        # Instantiate model.
        self.resnet = timm.create_model(timm_name, pretrained=pretrained)

        for bottleneck in self.resnet.layer3:
            bottleneck.act3 = torch.nn.Sequential()
        assert self.ho == 16 and self.wo == 16  # We have four downsampling operations.
        self.output_feature_dim = 1024  # layer3 output size = final embedding size.

        # Replace first convolutional layer to accommodate non-standard inputs.
        if in_channels != 3:
            assert not(pretrained)
            self.resnet.conv1 = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2),
                padding=(3, 3), bias=False)

    def forward(self, input_pixels):
        '''
        :param input_pixels (B, C, H, W) tensor.
        '''

        # Normalize if pretrained.
        if self.pretrained:
            mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=input_pixels.dtype,
                                device=input_pixels.device)
            mean = mean[:, None, None].expand_as(input_pixels[0])
            std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=input_pixels.dtype,
                               device=input_pixels.device)
            std = std[:, None, None].expand_as(input_pixels[0])
            input_pixels = input_pixels - mean
            input_pixels = input_pixels / std

        # Adapted from
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py.
        # Skip layer3 final ReLU, layer4, global_pool, flatten, and fc.
        x = self.resnet.conv1(input_pixels)
        x = self.resnet.bn1(x)
        x = self.resnet.act1(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        output_features = x  # (B, D, H, W).

        assert output_features.shape[1] == self.output_feature_dim

        return output_features


class MyDenseResNetBackbone(DenseResNet):
    '''
    Trainable variant of the DenseResNet.
    '''

    def __init__(self, logger, frame_height=224, frame_width=288, in_channels=3, pretrained=False):
        super().__init__(logger, 'resnet50', pretrained, frame_height, frame_width, 16, in_channels)
