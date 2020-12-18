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
import logging
import os

import torch.nn as nn
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

import cogan_pytorch.models as models
from cogan_pytorch.models import discriminator
import cogan_pytorch.datasets
from cogan_pytorch.utils import init_torch_seeds
from cogan_pytorch.utils import select_device
from cogan_pytorch.utils import weights_init

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Set random initialization seed, easy to reproduce.
        init_torch_seeds(args.manualSeed)

        logger.info("Load training dataset")
        # Selection of appropriate treatment equipment.
        input_dataset = torchvision.datasets.MNIST(root=args.dataroot, download=True,
                                                   transform=transforms.Compose([
                                                       transforms.Resize(args.image_size),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5,), (0.5,))
                                                   ]))
        target_dataset = cogan_pytorch.datasets.MNISTM(root=f"{args.dataroot}/MNIST", download=False,
                                                       transform=transforms.Compose([
                                                           transforms.Resize(args.image_size),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                       ]))
        self.input_dataloader = torch.utils.data.DataLoader(input_dataset,
                                                            batch_size=args.batch_size,
                                                            pin_memory=True,
                                                            num_workers=int(args.workers))
        self.target_dataloader = torch.utils.data.DataLoader(target_dataset,
                                                             batch_size=args.batch_size,
                                                             pin_memory=True,
                                                             num_workers=int(args.workers))

        logger.info(f"Train Dataset information:\n"
                    f"\tTrain Dataset dir is `{os.getcwd()}/{args.dataroot}`\n"
                    f"\tBatch size is {args.batch_size}\n"
                    f"\tWorkers is {int(args.workers)}\n"
                    f"\tLoad dataset to CUDA")

        # Construct network architecture model of generator and discriminator.
        self.device = select_device(args.device, batch_size=1)
        if args.pretrained:
            logger.info(f"Using pre-trained model `{args.arch}`")
            self.generator = models.__dict__[args.arch](pretrained=True,
                                                        image_size=args.image_size,
                                                        channels=args.channels).to(self.device)
        else:
            logger.info(f"Creating model `{args.arch}`")
            self.generator = models.__dict__[args.arch](image_size=args.image_size,
                                                        channels=args.channels).to(self.device)
        logger.info(f"Creating discriminator model")
        self.discriminator = discriminator(image_size=args.image_size, channels=args.channels).to(self.device)

        self.generator = self.generator.apply(weights_init)
        self.discriminator = self.discriminator.apply(weights_init)

        # Parameters of pre training model.
        self.epochs = int(int(args.iters) // len(self.input_dataloader))
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

        logger.info(f"Model training parameters:\n"
                    f"\tIters is {int(args.iters)}\n"
                    f"\tEpoch is {int(self.epochs)}\n"
                    f"\tOptimizer Adam\n"
                    f"\tLearning rate {args.lr}\n"
                    f"\tBetas (0.5, 0.999)")

        self.adversarial_criterion = nn.MSELoss().to(self.device)
        logger.info(f"Loss function:\n"
                    f"\tAdversarial loss is MSELoss")

    def run(self):
        args = self.args

        # Load pre training model.
        if args.netD != "":
            self.discriminator.load_state_dict(torch.load(args.netD))
        if args.netG != "":
            self.generator.load_state_dict(torch.load(args.netG))

        # Start train PSNR model.
        logger.info(f"Training for {self.epochs} epochs")

        fixed_noise = torch.randn(args.batch_size, 100, device=self.device)

        for epoch in range(args.start_epoch, self.epochs):
            iterable = enumerate(zip(self.input_dataloader, self.target_dataloader))
            progress_bar = tqdm(iterable, total=len(self.input_dataloader))
            for i, (data1, data2) in progress_bar:
                input = data1[0].to(self.device)
                target = data2[0].to(self.device)
                input = input.type(torch.Tensor).expand(input.size(0), args.channels, args.image_size, args.image_size)
                batch_size = input.size(0)
                real_label = torch.full((batch_size, 1), 1, dtype=input.dtype, device=self.device)
                fake_label = torch.full((batch_size, 1), 0, dtype=input.dtype, device=self.device)

                ##############################################
                # (1) Update D network: maximize - E(hr)[1- log(D(hr, sr))] - E(sr)[log(D(sr, hr))]
                ##############################################
                # train with real
                # Set discriminator gradients to zero.
                self.discriminator.zero_grad()

                input_output, target_output = self.discriminator(input, target)
                output = (input_output + target_output) / 2
                errD_input_real = self.adversarial_criterion(input_output, real_label)
                errD_target_real = self.adversarial_criterion(target_output, real_label)
                errD_real = errD_input_real + errD_target_real
                errD_real.backward()
                D_x = output.mean().item()

                # train with fake
                noise = torch.randn(batch_size, 100, device=self.device)
                fake1, fake2 = self.generator(noise)
                input_output, target_output = self.discriminator(fake1.detach(), fake2.detach())
                output = (input_output + target_output) / 2
                errD_input_fake = self.adversarial_criterion(input_output, fake_label)
                errD_target_fake = self.adversarial_criterion(target_output, fake_label)
                errD_fake = errD_input_fake + errD_target_fake
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                self.optimizer_d.step()

                ##############################################
                # (2) Update G network: maximize - E(hr)[log(D(hr, sr))] - E(sr)[1- log(D(sr, hr))]
                ##############################################
                # Set generator gradients to zero
                self.generator.zero_grad()

                input_output, target_output = self.discriminator(fake1, fake2)
                output = (input_output + target_output) / 2
                errG_input = self.adversarial_criterion(input_output, real_label)
                errG_target = self.adversarial_criterion(target_output, real_label)
                errG = errG_input + errG_target
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizer_g.step()

                progress_bar.set_description(f"[{epoch + 1}/{self.epochs}][{i + 1}/{len(self.input_dataloader)}] "
                                             f"Loss_D: {errD.item():.6f} Loss_G: {errG.item():.6f} "
                                             f"D(x): {D_x:.6f} D(G(z)): {D_G_z1:.6f}/{D_G_z2:.6f}")

                # The image is saved every 1 epoch.
                if (i + 1) % args.save_freq == 0:
                    vutils.save_image(input, os.path.join("output", "real_samples.bmp"))
                    fake1, fake2 = self.generator(fixed_noise)
                    fake = torch.cat([fake1.data, fake2.data], dim=0)
                    vutils.save_image(fake.detach(), os.path.join("output", f"fake_samples{epoch + 1}.bmp"))

            # do checkpointing
            torch.save(self.generator.state_dict(), f"weights/netG_epoch_{epoch + 1}.pth")
            torch.save(self.discriminator.state_dict(), f"weights/netD_epoch_{epoch + 1}.pth")
