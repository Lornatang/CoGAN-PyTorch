# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = [
    "Discriminator", "Generator", "discriminator",
    "mnist",
]

model_urls = {
    "mnist": "https://github.com/Lornatang/CoGAN-PyTorch/releases/download/0.1.0/CoGAN_mnist-845ad82d.pth",
}


class Discriminator(nn.Module):
    r""" An Discriminator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1606.07536v2>`_ paper.
    """

    def __init__(self, image_size: int = 32, channels: int = 3):
        """
        Args:
            image_size (int): The size of the image. (Default: 32).
            channels (int): The channels of the image. (Default: 3).
        """
        super(Discriminator, self).__init__()

        self.shared_conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.25)
        )

        # The height and width of down-sampled image
        ds_size = image_size // 2 ** 4
        self.branch1 = nn.Linear(128 * ds_size ** 2, 1)
        self.branch2 = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, input: torch.Tensor, target) -> [torch.Tensor, torch.Tensor]:
        r""" Defines the computation performed at every call.

        Args:
          input (tensor): Input tensor into the calculation.
          target (tensor): Target tensor into the calculation.

        Returns:
          Two four-dimensional vector (NCHW).
        """
        out1 = self.shared_conv(input)
        out1 = torch.flatten(out1, 1)
        out1 = self.branch1(out1)

        out2 = self.shared_conv(target)
        out2 = torch.flatten(out2, 1)
        out2 = self.branch2(out2)
        return out1, out2


class Generator(nn.Module):
    r""" An Generator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1606.07536v2>`_ paper.
    """

    def __init__(self, image_size: int = 32, channels: int = 3):
        """
        Args:
            image_size (int): The size of the image. (Default: 32).
            channels (int): The channels of the image. (Default: 3).
        """
        super(Generator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.init_size = image_size // 4
        self.fc = nn.Linear(100, 128 * self.init_size ** 2)

        self.shared_conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Upsample(scale_factor=2)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        r"""Defines the computation performed at every call.

        Args:
          input (tensor): input tensor into the calculation.

        Returns:
          A four-dimensional vector (NCHW).
        """
        out = self.fc(input)
        out = out.reshape(out.size(0), 128, self.init_size, self.init_size)

        out = self.shared_conv(out)
        out1 = self.branch1(out)
        out2 = self.branch2(out)
        return out1, out2


def _gan(arch, pretrained, progress, **kwargs):
    model = Generator(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def discriminator(**kwargs) -> Discriminator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1606.07536v2>`_ paper.
    """
    model = Discriminator(**kwargs)
    return model


def mnist(pretrained: bool = False, progress: bool = True, **kwargs) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1606.07536v2>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("mnist", pretrained, progress, **kwargs)
