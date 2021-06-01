# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
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


class DiscriminatorForMNIST(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network discriminator.

    Args:
        image_size (int): The size of the image. (Default: 32)
        channels (int): The channels of the image. (Default: 3)
    """

    def __init__(self, image_size: int = 32, channels: int = 3) -> None:
        super(DiscriminatorForMNIST, self).__init__()

        self.shared_conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(p=0.25, inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(p=0.25, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(p=0.25, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(p=0.25, inplace=True)
        )

        # The height and width of down-sampled image.
        ds_size = image_size // 2 ** 4
        self.branch1 = nn.Linear(128 * ds_size ** 2, 1)
        self.branch2 = nn.Linear(128 * ds_size ** 2, 1)

        # Initializing all neural network weights.
        self._initialize_weights()

    def forward(self, inputs: torch.Tensor, target) -> torch.Tensor:
        r""" Defines the computation performed at every call.

        Args:
          inputs (tensor): Input tensor into the calculation.
          target (tensor): Target tensor into the calculation.

        Returns:
          Two four-dimensional vector (N*C*H*W).
        """
        out1 = self.shared_conv(inputs)
        out1 = torch.flatten(out1, 1)
        out1 = self.branch1(out1)

        out2 = self.shared_conv(target)
        out2 = torch.flatten(out2, 1)
        out2 = self.branch2(out2)

        return out1, out2

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def discriminator_for_mnist(image_size: int = 32, channels: int = 3) -> DiscriminatorForMNIST:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/1606.07536v2>` paper.
    """
    model = DiscriminatorForMNIST(image_size, channels)

    return model
